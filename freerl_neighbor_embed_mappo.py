"""Torch reimplementation of the neighbour-embedding MAPPO trainer.

This module ports the original JAX-based script into a single-file PyTorch
version that follows the lightweight training-loop conventions used by the
FreeRL examples.  The script defines the coverage environment, policy/value
networks with recurrent neighbour encoders, as well as a minimal PPO training
loop that can be driven via a YAML configuration.
"""

from __future__ import annotations

import pathlib
import random
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import yaml


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def point_in_polygon(point: np.ndarray, verts: np.ndarray) -> bool:
    """Return ``True`` if the 2-D point lies inside the polygon."""

    x, y = point
    x0, y0 = verts[:, 0], verts[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    cond = ((y0 <= y) & (y < y1)) | ((y1 <= y) & (y < y0))
    xints = x0 + (y - y0) * (x1 - x0) / (y1 - y0 + 1e-8)
    hits = cond & (xints > x)
    return bool(np.sum(hits) % 2)


def rejection_sample_in_polygon(
    rng: np.random.Generator, verts: np.ndarray
) -> np.ndarray:
    """Sample a point uniformly inside a polygon using rejection sampling."""

    min_xy = verts.min(axis=0)
    max_xy = verts.max(axis=0)
    while True:
        sample = rng.uniform(min_xy, max_xy)
        if point_in_polygon(sample, verts):
            return sample


# ---------------------------------------------------------------------------
# Coverage environment
# ---------------------------------------------------------------------------


@dataclass
class CoverageMetrics:
    coverage_rate: float
    overlap_ratio: float
    unfairness: float
    agent_cover_share: np.ndarray


class SimpleCoverageEnv:
    """Multi-agent polygon coverage environment using NumPy operations."""

    def __init__(
        self,
        num_agents: int,
        polygon_vertices: Sequence[Sequence[float]],
        radius: float,
        max_steps: int,
        dt: float = 0.2,
        max_neighbors: Optional[int] = None,
        grid_size: int = 64,
    ) -> None:
        self.num_agents = num_agents
        self.polygon = np.asarray(polygon_vertices, dtype=np.float32)
        self.radius = float(radius)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.grid_size = int(grid_size)
        self.max_neighbors = max_neighbors or max(1, num_agents - 1)
        self.action_size = 5  # stay, up, down, left, right

        centroid = self.polygon.mean(axis=0)
        self.centroid = centroid.astype(np.float32)
        self.step_count = 0
        self.rng = np.random.default_rng()

        self.positions = np.zeros((num_agents, 2), dtype=np.float32)
        self.velocities = np.zeros((num_agents, 2), dtype=np.float32)
        self.metrics = CoverageMetrics(0.0, 0.0, 0.0, np.zeros(num_agents))

    # ------------------------------------------------------------------
    # Environment API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        for i in range(self.num_agents):
            self.positions[i] = rejection_sample_in_polygon(self.rng, self.polygon)
        self.velocities[:] = 0.0
        self.metrics = self._compute_metrics()
        obs = self._get_observations()
        return obs, {"coverage": self.metrics.coverage_rate}

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
        """Advance the environment by one step."""

        assert actions.shape == (self.num_agents,)
        actions = actions.astype(np.int64)
        deltas = np.array(
            [
                [0.0, 0.0],  # stay
                [0.0, 1.0],  # up
                [0.0, -1.0],  # down
                [-1.0, 0.0],  # left
                [1.0, 0.0],  # right
            ],
            dtype=np.float32,
        )
        accel = deltas[actions]
        self.velocities = 0.5 * self.velocities + accel
        self.positions = self.positions + self.dt * self.velocities
        self.positions = np.array(
            [self._project_inside_polygon(p) for p in self.positions], dtype=np.float32
        )

        self.step_count += 1
        self.metrics = self._compute_metrics()
        obs = self._get_observations()
        reward = self._compute_reward()
        done = np.array([self.step_count >= self.max_steps] * self.num_agents)
        info = {
            "coverage": self.metrics.coverage_rate,
            "overlap": self.metrics.overlap_ratio,
            "unfairness": self.metrics.unfairness,
        }
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Observation and reward helpers
    # ------------------------------------------------------------------
    def _project_inside_polygon(self, point: np.ndarray) -> np.ndarray:
        if point_in_polygon(point, self.polygon):
            return point
        # Simple projection: move towards centroid until inside
        direction = self.centroid - point
        for _ in range(10):
            point = point + 0.5 * direction
            if point_in_polygon(point, self.polygon):
                return point
        return self.centroid.copy()

    def _get_observations(self) -> np.ndarray:
        metrics = self.metrics
        local_feats = []
        neighbours = []
        fairness = metrics.unfairness
        overlap = metrics.overlap_ratio
        for i in range(self.num_agents):
            pos = self.positions[i]
            vel = self.velocities[i]
            dist_centre = np.linalg.norm(pos - self.centroid)
            cov_share = metrics.agent_cover_share[i]
            feats = np.array(
                [
                    pos[0],
                    pos[1],
                    vel[0],
                    vel[1],
                    dist_centre,
                    cov_share,
                    fairness,
                    overlap,
                ],
                dtype=np.float32,
            )
            local_feats.append(feats)

            others = []
            for j in range(self.num_agents):
                if i == j:
                    continue
                rel_pos = self.positions[j] - pos
                rel_vel = self.velocities[j] - vel
                others.append(np.concatenate([rel_pos, rel_vel]).astype(np.float32))
            while len(others) < self.max_neighbors:
                others.append(np.zeros(4, dtype=np.float32))
            neighbours.append(np.stack(others[: self.max_neighbors]))

        local = np.stack(local_feats)
        neigh = np.stack(neighbours)
        obs = np.concatenate([local, neigh.reshape(self.num_agents, -1)], axis=-1)
        return obs

    def _compute_reward(self) -> np.ndarray:
        metrics = self.metrics
        coverage_reward = metrics.coverage_rate
        overlap_penalty = metrics.overlap_ratio
        fairness_bonus = 1.0 - metrics.unfairness
        reward = coverage_reward + 0.2 * fairness_bonus - 0.1 * overlap_penalty
        return np.full((self.num_agents,), reward, dtype=np.float32)

    # ------------------------------------------------------------------
    # Coverage metrics
    # ------------------------------------------------------------------
    def _compute_metrics(self) -> CoverageMetrics:
        verts = self.polygon
        xs = np.linspace(verts[:, 0].min(), verts[:, 0].max(), self.grid_size)
        ys = np.linspace(verts[:, 1].min(), verts[:, 1].max(), self.grid_size)
        xv, yv = np.meshgrid(xs, ys)
        grid = np.stack([xv, yv], axis=-1).reshape(-1, 2)
        mask = np.array([point_in_polygon(p, verts) for p in grid], dtype=bool)
        inside_points = grid[mask]
        if inside_points.size == 0:
            return CoverageMetrics(0.0, 0.0, 0.0, np.zeros(self.num_agents))

        dist = np.linalg.norm(
            inside_points[:, None, :] - self.positions[None, :, :], axis=-1
        )
        covered = dist <= self.radius
        cover_counts = covered.sum(axis=1)
        coverage_rate = float(np.mean(cover_counts > 0))
        overlap_ratio = float(
            np.mean(cover_counts > 1) if np.any(cover_counts > 0) else 0.0
        )

        agent_cover = covered.sum(axis=0).astype(np.float32)
        total_cover = max(1.0, float(agent_cover.sum()))
        agent_share = agent_cover / total_cover
        unfairness = float(np.std(agent_share))
        return CoverageMetrics(coverage_rate, overlap_ratio, unfairness, agent_share)


# ---------------------------------------------------------------------------
# Vectorised environment wrapper
# ---------------------------------------------------------------------------


class CoverageVecEnv:
    def __init__(self, env_fns: Sequence[Callable[[], SimpleCoverageEnv]]):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.num_agents = self.envs[0].num_agents
        self.obs_dim = self.envs[0]._get_observations().shape[-1]
        self.max_neighbors = self.envs[0].max_neighbors

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        obs_batch = []
        for idx, env in enumerate(self.envs):
            env_seed = None if seed is None else seed + idx
            obs, _ = env.reset(env_seed)
            obs_batch.append(obs)
        return np.stack(obs_batch)  # (num_envs, num_agents, obs_dim)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        next_obs = []
        rewards = []
        dones = []
        coverages = []
        overlaps = []
        unfairness = []
        for env, env_actions in zip(self.envs, actions):
            obs, rew, done, info = env.step(env_actions)
            if done.any():
                obs_reset, _ = env.reset()
                done = np.ones_like(done)
                obs = obs_reset
            next_obs.append(obs)
            rewards.append(rew)
            dones.append(done)
            coverages.append(info["coverage"])
            overlaps.append(info["overlap"])
            unfairness.append(info["unfairness"])
        infos = {
            "coverage": np.array(coverages, dtype=np.float32),
            "overlap": np.array(overlaps, dtype=np.float32),
            "unfairness": np.array(unfairness, dtype=np.float32),
        }
        return (
            np.stack(next_obs),
            np.stack(rewards),
            np.stack(dones),
            infos,
        )


# ---------------------------------------------------------------------------
# Policy and value networks
# ---------------------------------------------------------------------------


class ObservationEmbedder(nn.Module):
    def __init__(self, local_dim: int, neighbor_dim: int, max_neighbors: int, embed_dim: int):
        super().__init__()
        self.local_dim = local_dim
        self.neighbor_dim = neighbor_dim
        self.max_neighbors = max_neighbors
        self.embed = nn.Linear(neighbor_dim, embed_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        local = obs[..., : self.local_dim]
        neigh_flat = obs[..., self.local_dim : self.local_dim + self.max_neighbors * self.neighbor_dim]
        neigh = neigh_flat.view(*obs.shape[:-1], self.max_neighbors, self.neighbor_dim)
        neigh_encoded = torch.relu(self.embed(neigh))
        neigh_agg = neigh_encoded.mean(dim=-2)
        return torch.cat([local, neigh_agg], dim=-1)


class ActorNetwork(nn.Module):
    def __init__(
        self,
        action_dim: int,
        local_dim: int,
        neighbor_dim: int,
        max_neighbors: int,
        embed_dim: int,
        hidden_dim: int,
        fc_dim: int,
    ) -> None:
        super().__init__()
        self.embed = ObservationEmbedder(local_dim, neighbor_dim, max_neighbors, embed_dim)
        self.fc = nn.Linear(local_dim + embed_dim, fc_dim)
        self.rnn = nn.GRUCell(fc_dim, hidden_dim)
        self.rnn_out = nn.Linear(hidden_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, action_dim)

    def forward(
        self, obs: torch.Tensor, hidden: torch.Tensor, dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = torch.relu(self.fc(self.embed(obs)))
        hidden = hidden * (1.0 - dones.unsqueeze(-1))
        hidden = self.rnn(embedded, hidden)
        features = torch.relu(self.rnn_out(hidden))
        logits = self.policy(features)
        return hidden, logits


class CriticNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, fc_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, fc_dim)
        self.rnn = nn.GRUCell(fc_dim, hidden_dim)
        self.rnn_out = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(
        self, obs: torch.Tensor, hidden: torch.Tensor, dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.fc(obs))
        hidden = hidden * (1.0 - dones.unsqueeze(-1))
        hidden = self.rnn(x, hidden)
        features = torch.relu(self.rnn_out(hidden))
        value = self.value(features).squeeze(-1)
        return hidden, value


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NeighborEmbedConfig:
    env_num_agents: int = 6
    env_polygon: Tuple[Tuple[float, float], ...] = (
        (-1.5, -1.0),
        (1.5, -1.0),
        (1.8, 0.5),
        (0.0, 1.5),
        (-1.8, 0.5),
    )
    env_radius: float = 0.5
    env_max_steps: int = 200
    env_grid_size: int = 96

    total_timesteps: int = 50_000
    num_envs: int = 4
    num_steps: int = 64
    num_minibatches: int = 4
    update_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    anneal_lr: bool = True

    gru_hidden_dim: int = 128
    fc_dim: int = 128
    embed_dim: int = 64
    local_dim: int = 8
    neighbor_dim: int = 4
    max_neighbors: Optional[int] = None

    seed: int = 42
    device: str = "cpu"

    @classmethod
    def from_yaml(cls, path: pathlib.Path) -> "NeighborEmbedConfig":
        raw = yaml.safe_load(path.read_text())
        mapping = {
            "lr": "lr",
            "num_envs": "num_envs",
            "num_steps": "num_steps",
            "total_timesteps": "total_timesteps",
            "fc_dim_size": "fc_dim",
            "gru_hidden_dim": "gru_hidden_dim",
            "update_epochs": "update_epochs",
            "num_minibatches": "num_minibatches",
            "gamma": "gamma",
            "gae_lambda": "gae_lambda",
            "clip_eps": "clip_eps",
            "ent_coef": "ent_coef",
            "vf_coef": "vf_coef",
            "max_grad_norm": "max_grad_norm",
            "anneal_lr": "anneal_lr",
            "seed": "seed",
            "embed_dim": "embed_dim",
            "local_dim": "local_dim",
            "max_neighbors": "max_neighbors",
        }
        int_fields = {
            "num_envs",
            "num_steps",
            "total_timesteps",
            "fc_dim",
            "gru_hidden_dim",
            "update_epochs",
            "num_minibatches",
            "max_neighbors",
            "seed",
            "embed_dim",
            "local_dim",
        }
        float_fields = {
            "lr",
            "gamma",
            "gae_lambda",
            "clip_eps",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
        }
        config_kwargs: Dict[str, object] = {}
        for key, value in raw.items():
            norm_key = key.lower()
            if norm_key in mapping:
                target = mapping[norm_key]
                if target in int_fields:
                    config_kwargs[target] = int(value)
                elif target in float_fields:
                    config_kwargs[target] = float(value)
                else:
                    config_kwargs[target] = value
        return cls(**config_kwargs)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------


@dataclass
class RolloutStorage:
    obs: torch.Tensor
    world_state: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    actor_h: torch.Tensor
    critic_h: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    @classmethod
    def init(cls, num_steps: int, batch_size: int, obs_dim: int, world_dim: int, hidden_dim: int, device: torch.device) -> "RolloutStorage":
        zeros_obs = torch.zeros(num_steps, batch_size, obs_dim, device=device)
        zeros_world = torch.zeros(num_steps, batch_size, world_dim, device=device)
        zeros_actions = torch.zeros(num_steps, batch_size, dtype=torch.long, device=device)
        zeros_logprobs = torch.zeros(num_steps, batch_size, device=device)
        zeros_rewards = torch.zeros(num_steps, batch_size, device=device)
        zeros_dones = torch.zeros(num_steps, batch_size, device=device)
        zeros_values = torch.zeros(num_steps, batch_size, device=device)
        zeros_hidden = torch.zeros(num_steps, batch_size, hidden_dim, device=device)
        zeros_advantages = torch.zeros(num_steps, batch_size, device=device)
        zeros_returns = torch.zeros(num_steps, batch_size, device=device)
        return cls(
            zeros_obs,
            zeros_world,
            zeros_actions,
            zeros_logprobs,
            zeros_rewards,
            zeros_dones,
            zeros_values,
            zeros_hidden.clone(),
            zeros_hidden.clone(),
            zeros_advantages,
            zeros_returns,
        )


# ---------------------------------------------------------------------------
# MAPPO trainer
# ---------------------------------------------------------------------------


class NeighborEmbedMAPPO:
    def __init__(self, config: NeighborEmbedConfig) -> None:
        self.cfg = config
        set_seed(config.seed)
        self.device = torch.device(config.device)

        env_fns = [
            lambda: SimpleCoverageEnv(
                num_agents=config.env_num_agents,
                polygon_vertices=config.env_polygon,
                radius=config.env_radius,
                max_steps=config.env_max_steps,
                grid_size=config.env_grid_size,
                max_neighbors=config.max_neighbors,
            )
            for _ in range(config.num_envs)
        ]
        self.envs = CoverageVecEnv(env_fns)

        self.num_agents = config.env_num_agents
        self.obs_dim = self.envs.obs_dim
        self.max_neighbors = self.envs.max_neighbors
        self.world_dim = self.obs_dim * self.num_agents
        self.batch_size = config.num_envs * config.env_num_agents

        max_neighbors = config.max_neighbors or self.max_neighbors
        self.actor = ActorNetwork(
            action_dim=self.envs.envs[0].action_size,
            local_dim=config.local_dim,
            neighbor_dim=config.neighbor_dim,
            max_neighbors=max_neighbors,
            embed_dim=config.embed_dim,
            hidden_dim=config.gru_hidden_dim,
            fc_dim=config.fc_dim,
        ).to(self.device)
        self.critic = CriticNetwork(
            input_dim=self.world_dim,
            hidden_dim=config.gru_hidden_dim,
            fc_dim=config.fc_dim,
        ).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.lr)

        self._obs_np = self.envs.reset(config.seed)
        self._prev_dones = torch.zeros(self.batch_size, device=self.device)
        self._actor_hidden = torch.zeros(
            self.batch_size, config.gru_hidden_dim, device=self.device
        )
        self._critic_hidden = torch.zeros_like(self._actor_hidden)

    # ------------------------------------------------------------------
    def collect_rollouts(self) -> RolloutStorage:
        cfg = self.cfg
        num_steps = cfg.num_steps
        device = self.device
        obs = torch.as_tensor(self._obs_np, dtype=torch.float32, device=device)
        obs = obs.view(cfg.num_envs * self.num_agents, self.obs_dim)

        actor_hidden = self._actor_hidden.detach().clone()
        critic_hidden = self._critic_hidden.detach().clone()
        prev_dones = self._prev_dones.clone()

        storage = RolloutStorage.init(
            num_steps,
            self.batch_size,
            self.obs_dim,
            self.world_dim,
            cfg.gru_hidden_dim,
            device,
        )

        for step in range(num_steps):
            world_state = obs.view(cfg.num_envs, self.num_agents, -1)
            world_state = world_state.reshape(cfg.num_envs, -1)
            world_state = world_state.repeat_interleave(self.num_agents, dim=0)

            actor_h_in = actor_hidden
            critic_h_in = critic_hidden

            actor_hidden, logits = self.actor(obs, actor_h_in, prev_dones)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            critic_hidden, values = self.critic(world_state, critic_h_in, prev_dones)

            storage.obs[step] = obs
            storage.world_state[step] = world_state
            storage.actions[step] = actions
            storage.log_probs[step] = log_probs
            storage.values[step] = values
            storage.actor_h[step] = actor_h_in
            storage.critic_h[step] = critic_h_in

            actions_env = actions.view(cfg.num_envs, self.num_agents).cpu().numpy()
            next_obs_np, rewards_np, dones_np, _ = self.envs.step(actions_env)

            rewards = torch.as_tensor(
                rewards_np.reshape(-1), dtype=torch.float32, device=device
            )
            dones_tensor = torch.as_tensor(
                dones_np.reshape(-1), dtype=torch.float32, device=device
            )

            storage.rewards[step] = rewards
            storage.dones[step] = dones_tensor

            obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            obs = obs.view(cfg.num_envs * self.num_agents, self.obs_dim)
            prev_dones = dones_tensor

            actor_hidden = actor_hidden.detach()
            critic_hidden = critic_hidden.detach()

        self._obs_np = next_obs_np
        self._prev_dones = prev_dones.detach()
        self._actor_hidden = actor_hidden.detach()
        self._critic_hidden = critic_hidden.detach()

        with torch.no_grad():
            world_state = obs.view(cfg.num_envs, self.num_agents, -1)
            world_state = world_state.reshape(cfg.num_envs, -1)
            world_state = world_state.repeat_interleave(self.num_agents, dim=0)
            _, next_value = self.critic(world_state, critic_hidden, prev_dones)

        self._compute_gae(storage, next_value)
        return storage

    # ------------------------------------------------------------------
    def _compute_gae(self, storage: RolloutStorage, next_value: torch.Tensor) -> None:
        cfg = self.cfg
        advantages = torch.zeros_like(storage.rewards)
        last_adv = torch.zeros(self.batch_size, device=self.device)
        for t in reversed(range(cfg.num_steps)):
            next_non_terminal = 1.0 - storage.dones[t]
            if t == cfg.num_steps - 1:
                next_values = next_value
            else:
                next_values = storage.values[t + 1]
            delta = (
                storage.rewards[t]
                + cfg.gamma * next_values * next_non_terminal
                - storage.values[t]
            )
            last_adv = (
                delta
                + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_adv
            )
            advantages[t] = last_adv
        storage.advantages[:] = advantages
        storage.returns[:] = advantages + storage.values

    # ------------------------------------------------------------------
    def update(self, storage: RolloutStorage, update: int, total_updates: int) -> Dict[str, float]:
        cfg = self.cfg
        batch_size = cfg.num_steps * self.batch_size
        batch_indices = torch.randperm(batch_size)

        obs = storage.obs.reshape(batch_size, self.obs_dim)
        world_state = storage.world_state.reshape(batch_size, self.world_dim)
        actions = storage.actions.reshape(batch_size)
        old_log_probs = storage.log_probs.reshape(batch_size)
        advantages = storage.advantages.reshape(batch_size)
        returns = storage.returns.reshape(batch_size)
        values = storage.values.reshape(batch_size)
        actor_h = storage.actor_h.reshape(batch_size, cfg.gru_hidden_dim)
        critic_h = storage.critic_h.reshape(batch_size, cfg.gru_hidden_dim)
        dones = storage.dones.reshape(batch_size)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss_epoch = 0.0
        value_loss_epoch = 0.0
        entropy_epoch = 0.0

        minibatch_size = batch_size // cfg.num_minibatches
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_idx = batch_indices[start:end]

            mb_obs = obs[mb_idx]
            mb_world = world_state[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]
            mb_actor_h = actor_h[mb_idx]
            mb_critic_h = critic_h[mb_idx]
            mb_dones = dones[mb_idx]

            _, new_logits = self.actor(mb_obs, mb_actor_h, mb_dones)
            new_dist = Categorical(logits=new_logits)
            new_log_probs = new_dist.log_prob(mb_actions)
            entropy = new_dist.entropy().mean()

            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            pg_loss_1 = ratio * mb_advantages
            pg_loss_2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_advantages
            policy_loss = -torch.min(pg_loss_1, pg_loss_2).mean()

            _, new_values = self.critic(mb_world, mb_critic_h, mb_dones)
            value_loss = 0.5 * (mb_returns - new_values).pow(2).mean()

            self.actor_opt.zero_grad()
            actor_loss = policy_loss - cfg.ent_coef * entropy
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
            self.actor_opt.step()

            self.critic_opt.zero_grad()
            critic_loss = cfg.vf_coef * value_loss
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
            self.critic_opt.step()

            policy_loss_epoch += policy_loss.item()
            value_loss_epoch += value_loss.item()
            entropy_epoch += entropy.item()

        if cfg.anneal_lr:
            frac = (update + 1) / float(total_updates)
            lr_now = cfg.lr * (1.0 - frac)
            for param_group in self.actor_opt.param_groups:
                param_group["lr"] = lr_now
            for param_group in self.critic_opt.param_groups:
                param_group["lr"] = lr_now

        minibatches = cfg.num_minibatches
        return {
            "policy_loss": policy_loss_epoch / minibatches,
            "value_loss": value_loss_epoch / minibatches,
            "entropy": entropy_epoch / minibatches,
        }

    # ------------------------------------------------------------------
    def train(self) -> None:
        cfg = self.cfg
        total_updates = cfg.total_timesteps // (cfg.num_steps * cfg.num_envs)
        for update in range(total_updates):
            storage = self.collect_rollouts()
            stats = self.update(storage, update, total_updates)
            if (update + 1) % 10 == 0:
                print(
                    f"Update {update + 1}/{total_updates} :: "
                    f"policy_loss={stats['policy_loss']:.3f} "
                    f"value_loss={stats['value_loss']:.3f} "
                    f"entropy={stats['entropy']:.3f}"
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(config_path: Optional[str] = None) -> None:
    if config_path is None:
        cfg = NeighborEmbedConfig()
    else:
        cfg = NeighborEmbedConfig.from_yaml(pathlib.Path(config_path))
    trainer = NeighborEmbedMAPPO(cfg)
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neighbor-embed MAPPO trainer")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
