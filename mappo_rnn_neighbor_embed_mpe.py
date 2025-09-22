"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.
Integrated with ObservationEmbedder for neighbor observations using flat vector format.
"""
import sys
sys.path.insert(0, 'E:/SmallRepos/JaxMARL/') 

import os
import orbax.checkpoint as ocp
import orbax.checkpoint.args as ocp_args
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict
import chex

from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial
import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper, JaxMARLWrapper
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State

import wandb
import functools
import matplotlib.pyplot as plt
import time

    
class MPEWorldStateWrapper(JaxMARLWrapper):
    
    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state
    
    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(
            key, state, action
        )
        obs["world_state"] = self.world_state(obs)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def world_state(self, obs):
        """ 
        For each agent: [agent obs, all other agent obs]
        所谓的世界状态就是把所有agent的观测拼接起来展平，然后重复num_agents次
        
        供给Critic的输入
        """
        
        @partial(jax.vmap, in_axes=(0, None))
        def _roll_obs(aidx, all_obs):
            robs = jnp.roll(all_obs, -aidx, axis=0)
            robs = robs.flatten()
            return robs
            
        all_obs = jnp.array([obs[agent] for agent in self._env.agents]).flatten()
        all_obs = jnp.expand_dims(all_obs, axis=0).repeat(self._env.num_agents, axis=0)
        return all_obs
    
    def world_state_size(self):
        spaces = [self._env.observation_space(agent) for agent in self._env.agents]
        return sum([space.shape[-1] for space in spaces])


# ---自定义网络结构---
class ObservationEmbedder(nn.Module):
    """
    观测嵌入模块，接收平铺向量并内部切片为local和neighbors。
    """
    embed_dim: int
    config: Dict[str, Any]

    @nn.compact
    def __call__(self, flat_obs: jnp.ndarray) -> jnp.ndarray:
        # 从配置获取维度
        L = self.config["LOCAL_DIM"]      # 自身特征长度，一定要与环境一致
        D = self.config["NEI_FEAT_DIM"]   # 单邻居特征
        Nm = self.config["MAX_NEIGHBORS"] # 最大邻居数，要注意多个位置中MAX_NEIGHBORS的值一定要一样！环境中、配置中以及环境默认参数
        
        # 切片
        local_obs = flat_obs[..., :L]  # [1, NUM_ENVS, dim_local]
        neighbors_flat = flat_obs[..., L:L+Nm*D]  # [..., Nm*D]
        neighbors_obs = neighbors_flat.reshape(neighbors_flat.shape[:-1] + (Nm, D))  # [..., Nm, D]
        
        # 1. 邻居特征编码
        h = nn.Dense(self.embed_dim)(neighbors_obs)  # [..., Nm, E]
        h = nn.relu(h)
        
        # 2. 均值汇聚 (沿邻居维度)
        h_mean = jnp.mean(h, axis=-2)  # [..., E]
        
        # 3. 拼接本体观测
        return jnp.concatenate([local_obs, h_mean], axis=-1)  # [..., L+E]


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x  # obs现在是平铺向量
        
        # 使用ObservationEmbedder处理观测
        obs_embed = ObservationEmbedder(
            embed_dim=self.config["EMBED_DIM"],
            config=self.config
        )(obs)
        
        # 第一层全连接
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs_embed)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # actor_mean是Actor的中间特征表示层，从RNN输出的embedding进一步提取的高级特征表示
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        # 共享表征学习，因为Actor_mean已经表征了智能体的整体状态，因此可以直接用来决定所有维度上的动作，没有必要每个维度都配一个特征提取
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi  # 不再返回嵌入观测


class CriticRNN(nn.Module):
    config: Dict
    
    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x  # 接收world_state而不是嵌入观测
        
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        embedding = nn.relu(embedding)
        
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        critic = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        
        return hidden, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray  # 平铺向量 (Actor使用)
    world_state: jnp.ndarray  # world_state (Critic使用)
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    """将多智能体观测堆叠为批次"""
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    """将扁平化的动作转换回环境期望的格式"""
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    # 配置项
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]

    # 确保配置中有必要的维度信息
    config.setdefault("LOCAL_DIM", 8)  # 来自环境的local_obs_dim
    config.setdefault("EMBED_DIM", 128)
    config.setdefault("NEI_FEAT_DIM", 2)
    config.setdefault("MAX_NEIGHBORS", 9)  # 来自环境的max_neighbors

    env = MPEWorldStateWrapper(env)
    env = MPELogWrapper(env)

    # 学习率衰减器
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticRNN(config=config)
        
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        # 准备网络初始化输入
        # 获取实际的观测空间维度
        sample_obs, _ = env.reset(jax.random.PRNGKey(0))
        obs_dim = sample_obs[env.agents[0]].shape[-1]  # 平铺向量维度
        world_state_dim = env.world_state_size()  # world_state维度
        
        # Actor初始化输入
        ac_init_obs = jnp.zeros((1, config["NUM_ENVS"], obs_dim))
        ac_init_x = (ac_init_obs, jnp.zeros((1, config["NUM_ENVS"])))
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)
        
        # Critic初始化输入 - 使用world_state维度
        cr_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], world_state_dim)),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)
        
        # 优化器配置
        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            
        # 训练状态
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            
            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, hstates, rng = runner_state                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])  # 直接堆叠平铺向量
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                ac_hstate, pi = actor_network.apply(train_states[0].params, hstates[0], ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                
                # VALUE - 使用world_state而不是嵌入观测
                world_state = last_obs["world_state"].swapaxes(0,1).reshape((config["NUM_ACTORS"],-1))
                cr_in = (
                    world_state[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                cr_hstate, value = critic_network.apply(train_states[1].params, hstates[1], cr_in)                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    info,
                )
                runner_state = (train_states, env_state, obsv, done_batch, (ac_hstate, cr_hstate), rng)
                return runner_state, transition

            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, hstates, rng = runner_state
              # 计算最后一个状态的价值
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            last_world_state = last_obs["world_state"].swapaxes(0,1).reshape((config["NUM_ACTORS"],-1))
            
            cr_in = (
                last_world_state[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            _, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = batch_info
                    
                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        # RERUN NETWORK
                        _, pi = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()
                        
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        
                        actor_loss = (
                            loss_actor
                            - config["ENT_COEF"] * entropy
                        )
                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)
                    
                    def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                        # RERUN NETWORK - 使用world_state
                        _, value = critic_network.apply(
                            critic_params, 
                            init_hstate.squeeze(), 
                            (traj_batch.world_state, traj_batch.done)
                        )
                        
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params, ac_init_hstate, traj_batch, advantages
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, cr_init_hstate, traj_batch, targets
                    )
                    
                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
                    
                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                    }
                    
                    return (actor_train_state, critic_train_state), loss_info

                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstates = jax.tree.map(lambda x: jnp.reshape(
                    x, (1, config["NUM_ACTORS"], -1)
                ), init_hstates)
                
                batch = (
                    init_hstates[0],
                    init_hstates[1],
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_states, loss_info = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    jax.tree.map(lambda x: x.squeeze(), init_hstates),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info_epoch = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            
            final_loss_info = jax.tree.map(lambda x: x[-1].mean() if x.ndim > 1 else x[-1], loss_info_epoch)
            
            current_train_states = update_state[0]
            current_rng = update_state[-1]

            # 准备指标
            metrics_for_log = {
                "returned_episode_returns": traj_batch.info["returned_episode_returns"],
                "update_steps": update_steps,
                "loss": final_loss_info
            }
            if "returned_episode_lengths" in traj_batch.info:
                 metrics_for_log["returned_episode_lengths"] = traj_batch.info["returned_episode_lengths"]

            # 模型保存回调
            def _save_model_callback(actor_p, critic_p, step):
                if config.get("SAVE_MODEL", False) and step > 0 and step % config.get("SAVE_MODEL_INTERVAL", 100) == 0:
                    base_save_dir = config.get("MODEL_SAVE_PATH", "./jaxmarl_checkpoints")
                    absolute_base_save_dir = os.path.abspath(base_save_dir)

                    run_folder_name = config.get("GENERATED_RUN_NAME")
                    if not run_folder_name:
                        if wandb.run and wandb.run.name:
                            run_folder_name = wandb.run.name
                        else:
                            run_folder_name = f"marl_run_seed_{config['SEED']}_fallback_{int(time.time())}"
                    
                    manager_dir = os.path.join(absolute_base_save_dir, run_folder_name)
                    save_items = {'actor_params': actor_p, 'critic_params': critic_p}

                    options = ocp.CheckpointManagerOptions(
                        max_to_keep=config.get("MAX_CHECKPOINTS_TO_KEEP", None),
                        create=True 
                    )
                    
                    mngr = ocp.CheckpointManager(
                        manager_dir,
                        options=options
                    )

                    save_args = ocp_args.PyTreeSave(save_items)
                    mngr.save(step, args=save_args)
                    mngr.wait_until_finished()
                    
                    checkpoint_step_dir = os.path.join(manager_dir, str(step))
                    print(f"Orbax checkpoint for step {step} saved in directory: {checkpoint_step_dir}")

            actor_params_to_save = current_train_states[0].params
            critic_params_to_save = current_train_states[1].params
            jax.experimental.io_callback(
                _save_model_callback, None, actor_params_to_save, critic_params_to_save, update_steps
            )

            # W&B 日志回调
            def _wandb_log_callback(metrics):
                log_payload = {
                    "returns_mean": metrics["returned_episode_returns"][-1].mean(),
                    "env_step": metrics["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"],
                }
                if "returned_episode_lengths" in metrics:
                    log_payload["episode_length_mean"] = metrics["returned_episode_lengths"][-1].mean()
                
                for k, v in metrics["loss"].items():
                    log_payload[f"loss/{k}"] = v
                
                wandb.log(log_payload)
                            
            jax.experimental.io_callback(_wandb_log_callback, None, metrics_for_log)

            # 打印进度回调
            def _print_progress_callback(metrics_to_print, current_step, total_steps):
                if current_step % config.get("PRINT_INTERVAL_UPDATES", 20) == 0 or current_step == total_steps -1:
                    print_str = f"Update Step: {current_step}/{total_steps} | "
                    print_str += f"Env Steps: {current_step * config['NUM_ENVS'] * config['NUM_STEPS']} | "
                    
                    mean_return = metrics_to_print["returned_episode_returns"][-1].mean()
                    print_str += f"Mean Return: {mean_return:.2f} | "
                    if "returned_episode_lengths" in metrics_to_print:
                        mean_length = metrics_to_print["returned_episode_lengths"][-1].mean()
                        print_str += f"Mean Ep Length: {mean_length:.1f} | "
                    
                    loss_dict = metrics_to_print["loss"]
                    for loss_name, loss_val in loss_dict.items():
                        print_str += f"{loss_name}: {loss_val:.4f} | "
                    
                    print(print_str.strip().strip("|").strip())

            jax.experimental.io_callback(
                _print_progress_callback, None, metrics_for_log, update_steps, config["NUM_UPDATES"]
            )
            
            next_update_steps = update_steps + 1
            runner_state_for_next_iter = (current_train_states, env_state, last_obs, last_done, hstates, current_rng)
            
            return (runner_state_for_next_iter, next_update_steps), metrics_for_log

        rng, _rng = jax.random.split(rng)
        initial_runner_state_tuple = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            _rng,
        )
        
        (final_runner_state_tuple, final_update_steps_val), collected_metrics = jax.lax.scan(
            _update_step, (initial_runner_state_tuple, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": final_runner_state_tuple, "collected_metrics": collected_metrics}

    return train

@hydra.main(version_base=None, config_path="config", config_name="mappo_homogenous_rnn_mpe_embed")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)

    run_name = f"{config['ENV_NAME']}_{config.get('ALGO_NAME', 'MAPPO_RNN_EMBED')}_seed{config['SEED']}_{wandb.util.generate_id()}"
    config["GENERATED_RUN_NAME"] = run_name

    wandb.init(
        entity=config.get("ENTITY"),
        project=config.get("PROJECT"),
        tags=["MAPPO", "RNN", "EMBED", config["ENV_NAME"], config.get("TAG", "default_tag")],
        config=config,
        mode=config.get("WANDB_MODE", "online"),
        name=run_name,
        save_code=True,
    )

    if config.get("SAVE_MODEL", False):
        base_save_dir = config.get("MODEL_SAVE_PATH", "./jaxmarl_checkpoints")
        os.makedirs(base_save_dir, exist_ok=True) 
        print(f"模型保存已启用。基础保存目录: {os.path.abspath(base_save_dir)}")
        wandb.config.update({
            "SAVE_MODEL_ENABLED": True,
            "SAVE_MODEL_INTERVAL_UPDATES": config.get("SAVE_MODEL_INTERVAL", 100),
            "MODEL_CHECKPOINT_PATH": os.path.abspath(base_save_dir),
            "MAX_CHECKPOINTS_TO_KEEP": config.get("MAX_CHECKPOINTS_TO_KEEP", "All (None)"),
            "RUN_SAVE_FOLDER_NAME": config["GENERATED_RUN_NAME"]
        }, allow_val_change=True)
    else:
        wandb.config.update({"SAVE_MODEL_ENABLED": False}, allow_val_change=True)

    rng = jax.random.PRNGKey(config["SEED"])
    
    train_fn = make_train(config)
    
    if config.get("DISABLE_JIT", False):
        train_jit = train_fn
        print("JIT 编译已禁用 (用于调试).")
    else:
        train_jit = jax.jit(train_fn)
        print("JIT 编译已启用.")
        
    out = train_jit(rng)

    print("训练完成。")
    wandb.finish()
    
if __name__=="__main__":
    main()