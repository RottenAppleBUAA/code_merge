import sys
sys.path.insert(0, 'E:/SmallRepos/JaxMARL/')

import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid  # 用于二维梯形积分
import numpy as np  # 保留，以防需要 numpy 类型转换
from typing import Tuple, Dict, Optional
from functools import partial
from shapely.geometry import Polygon  # 仅用于存储多边形边界
from jaxmarl.environments.mpe.simple import SimpleMPE, State
from jaxmarl.environments.mpe.default_params import *
from jaxmarl.environments.spaces import Box
import chex


def point_in_poly(point: jnp.ndarray, verts: jnp.ndarray) -> jnp.ndarray:
    """
    使用射线法判断点是否在多边形内。

    参数:
      point: (2,) 点坐标
      verts: (M,2) 多边形顶点
    返回:
      0 或 1 的布尔值 jnp.bool_
    """
    x, y = point[0], point[1]
    x0, y0 = verts[:,0], verts[:,1]
    x1, y1 = jnp.roll(x0, -1), jnp.roll(y0, -1)
    # 条件: 边的两个端点的 y 值是否跨过目标点 y
    cond = ((y0 <= y) & (y < y1)) | ((y1 <= y) & (y < y0))
    # 计算射线与边相交的 x 坐标
    xints = x0 + (y - y0) * (x1 - x0) / (y1 - y0 + 1e-8)
    # 判断相交点是否在点的右侧
    inside = jnp.mod(jnp.sum(cond & (xints > x)), 2)
    return inside.astype(jnp.bool_)

def dist_to_segment(p: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    点 p 到线段 ab 的最短距离。
    p: (2,), a: (2,), b: (2,)
    """
    ap = p - a
    ab = b - a
    t = jnp.dot(ap, ab) / (jnp.dot(ab, ab) + 1e-8)
    t_clamped = jnp.clip(t, 0.0, 1.0)
    proj = a + t_clamped * ab
    return jnp.linalg.norm(p - proj)

def dist_to_poly(point: jnp.ndarray, verts: jnp.ndarray) -> jnp.ndarray:
    """
    点到多边形边界的最短距离。
    point: (2,), verts: (M,2)
    """
    a = verts
    b = jnp.roll(verts, -1, axis=0)
    dists = jax.vmap(lambda aa, bb: dist_to_segment(point, aa, bb))(a, b)
    return jnp.min(dists)

def closest_point_on_poly_relative(point: jnp.ndarray, verts: jnp.ndarray) -> jnp.ndarray:
    """
    计算点到多边形边界的最近点的相对位移向量。
    
    参数:
      point: (2,) 查询点坐标
      verts: (M,2) 多边形顶点
    
    返回:
      relative_vector: (2,) 从查询点指向多边形边界最近点的向量
    """
    
    def closest_point_on_segment(p: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """计算点p到线段ab的最近点"""
        ap = p - a
        ab = b - a
        
        ab_dot_ab = jnp.dot(ab, ab)
        t = jnp.where(ab_dot_ab > 1e-8, 
                     jnp.dot(ap, ab) / ab_dot_ab, 
                     0.0)
        
        t_clamped = jnp.clip(t, 0.0, 1.0)
        closest = a + t_clamped * ab
        return closest

    # 获取所有边的端点
    a_verts = verts                           
    b_verts = jnp.roll(verts, -1, axis=0)     

    # 计算点到每条边的最近点
    closest_points = jax.vmap(lambda a, b: closest_point_on_segment(point, a, b))(a_verts, b_verts)
    
    # 计算点到每个最近点的距离
    distances = jax.vmap(lambda cp: jnp.linalg.norm(point - cp))(closest_points)
    
    # 找到距离最小的点
    min_idx = jnp.argmin(distances)
    closest_point = closest_points[min_idx]
    
    # 返回相对位移向量：从查询点指向最近点
    relative_vector = closest_point - point
    
    return relative_vector

def compute_coverage_overlap(
    positions: jnp.ndarray,
    poly_verts: jnp.ndarray,
    r: float,
    grid_size: int = 128,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    通过网格采样 & 数值积分计算多边形内部的覆盖率 C、重叠率 R 和不公平度 U。

    positions: (N_agents,2)，agent 坐标
    poly_verts: (M,2)，多边形顶点
    r: 覆盖半径
    grid_size: 网格分辨率

    返回:
      C: 全局涵盖率 (标量)
      R_agents: 每个 agent 的重叠率 (N_agents,)
      R_global_orig: 原始定义的全局重叠率 (标量)
      U_norm: 归一化不公平度 (标量)
    """
    # 构建均匀采样网格
    x_min, x_max = jnp.min(poly_verts[:,0]), jnp.max(poly_verts[:,0])
    y_min, y_max = jnp.min(poly_verts[:,1]), jnp.max(poly_verts[:,1])
    xs = jnp.linspace(x_min, x_max, grid_size)
    ys = jnp.linspace(y_min, y_max, grid_size)
    X, Y = jnp.meshgrid(xs, ys, indexing='xy')  # shape (grid, grid)
    pts = jnp.stack([X.ravel(), Y.ravel()], axis=1)  # shape (grid^2, 2)

    # 判断每个采样点是否在多边形内部
    inside = jax.vmap(lambda p: point_in_poly(p, poly_verts))(pts)  # (grid^2,)

    # 计算每个点到所有 agent 的距离，并判定是否被覆盖
    dists = jnp.linalg.norm(pts[:,None,:] - positions[None,:,:], axis=-1)  # (grid^2, N)
    covered = dists < r  # (grid^2, N)
    # 仅保留多边形内部的覆盖点
    covered_in = covered & inside[:,None]  # (grid^2, N) (boolean)
    covered_in_float = covered_in.astype(jnp.float32) # (grid^2, N) (float for integration)

    # # 计算覆盖率 C = ∫ any(covered) dA / ∫1 dA
    # cov_mask_bool = jnp.any(covered_in, axis=1) # (grid^2,)
    # cov_mask_float = cov_mask_bool.astype(jnp.float32).reshape((grid_size,grid_size))
    # area_box = (x_max - x_min) * (y_max - y_min)
    # # 先沿 x 方向积分，再沿 y 方向积分
    # integral_cov_mask = trapezoid(trapezoid(cov_mask_float, xs, axis=1), ys, axis=0)
    # C = integral_cov_mask / area_box

    #! 覆盖率C的计算方式进行了修改，采用CAP与MCR的结合
    alpha = 0.5  # CAP与MCR的权重系数，目前是两者平衡
    # 当前时刻的覆盖掩膜
    current_coverage_mask = jnp.any(covered_in, axis=1) # (grid^2,)
    current_coverage_mask_2d = current_coverage_mask.astype(jnp.float32).reshape((grid_size, grid_size))
    # 计算当前覆盖面积 l(T ∩ P_t)
    integral_current_coverage = trapezoid(trapezoid(current_coverage_mask_2d, xs, axis=1), ys, axis=0)
    
    # 计算任务区域总面积 l(T)
    task_area_mask = inside.astype(jnp.float32).reshape((grid_size, grid_size))
    task_area = trapezoid(trapezoid(task_area_mask, xs, axis=1), ys, axis=0)
    
    # 计算CAP = l(T ∩ P_t) / l(T) × 100%
    CAP = jnp.where(task_area > 0, integral_current_coverage / task_area, 0.0)
    
    # 计算MCR
    N_agents = positions.shape[0]
    # 理论可用足迹上界 l_max = min(l(T), N*π*r²)
    theoretical_max_footprint = N_agents * jnp.pi * (r ** 2)
    l_max = jnp.minimum(task_area, theoretical_max_footprint)
    
    # MCR = l(T ∩ P_t) / l_max × 100%
    MCR = jnp.where(l_max > 0, integral_current_coverage / l_max, 0.0)
    
    # 组合覆盖率 C = alpha * CAP + (1 - alpha) * MCR
    C = alpha * CAP + (1.0 - alpha) * MCR

    # 计算原始全局重叠率 R_global_orig = ∫ count>1 dA / ∫ any dA (for bonus and done condition)
    count = jnp.sum(covered_in, axis=1)  # (grid^2,) 每个采样点被覆盖的次数
    original_overlap_mask_bool = (count > 1) # (grid^2,)
    original_overlap_mask_float = original_overlap_mask_bool.astype(jnp.float32).reshape((grid_size,grid_size))
    numer_global_R = trapezoid(trapezoid(original_overlap_mask_float, xs, axis=1), ys, axis=0)
    denom_global_R = integral_current_coverage # This is ∫ any(covered) dA
    R_global_orig = jnp.where(denom_global_R > 0, numer_global_R / denom_global_R, 0.0)

    # ---------- 新增：独占掩膜 ----------
    unique_flat = covered_in_float * (count[:, None] == 1).astype(jnp.float32)
    # (grid^2, N)  只被某个 agent 覆盖的采样点置 1

    # ---------- 已有：面积工具 ----------
    def calculate_area_from_flat_mask(flat_mask_float):
        reshaped_mask = flat_mask_float.reshape((grid_size, grid_size))
        return trapezoid(trapezoid(reshaped_mask, xs, axis=1), ys, axis=0)

    # ---------- 每 agent 面积 ----------
    A_total   = jax.vmap(calculate_area_from_flat_mask, in_axes=1, out_axes=0)(covered_in_float)
    A_unique  = jax.vmap(calculate_area_from_flat_mask, in_axes=1, out_axes=0)(unique_flat)

    # 计算每个 agent 的重叠率 R_i = (Area of overlap involving agent i) / (Area covered by agent i)
    N_agents = positions.shape[0]

    # 2. 计算每个 agent 的重叠面积 (numer_i)
    # A point contributes to agent i's overlap numerator if:
    #   a) agent i covers it (covered_in_float[:, i])
    #   b) the point is globally overlapped (original_overlap_mask_bool)
    agent_specific_overlap_contribution_masks_flat_float = covered_in_float * original_overlap_mask_bool[:, None].astype(jnp.float32)
    # agent_specific_overlap_contribution_masks_flat_float is (grid_size^2, N_agents)

    agent_specific_overlap_numerators = jax.vmap(calculate_area_from_flat_mask, in_axes=1, out_axes=0)(agent_specific_overlap_contribution_masks_flat_float)
    # agent_specific_overlap_numerators is (N_agents,)

    R_agents = jnp.where(A_total > 0, agent_specific_overlap_numerators / A_total, 0.0)

    # ---------- Jain 指数 ----------
    sum_Au   = jnp.sum(A_unique)
    fairness = jnp.where(sum_Au > 0,
                         (sum_Au**2) / (N_agents * jnp.sum(A_unique**2) + 1e-8),
                         0.0)
    U = 1.0 - fairness                                    # 不公平度 0~(1-1/N)
    # 归一化到 [0,1] 以完全对齐 R 的量级
    U_norm = U * (N_agents / (N_agents - 1.0 + 1e-8))

    return C, CAP, MCR, U_norm, R_agents


class SimpleCoverageMPE6(SimpleMPE):
    """
    使用纯 JAX 几何计算的覆盖任务环境，适配 MAPPO 等强化学习算法。
    覆盖率与重叠率计算均采用数值积分，无须依赖 Shapely。
    """
    def __init__(
        self,
        polygon_vertices: np.ndarray = np.array([
            [0.0,0.0], [10.0,0.0], [10.0,10.0], [0.0,10.0]
        ], dtype=np.float32),
        # polygon_vertices: np.ndarray = np.array([
        #     [1.0, 2.0], [8.5, 0.5], [12.0, 4.0], [10.5, 9.5], 
        #     [6.0, 11.0], [2.0, 8.5], [0.5, 5.0]
        # ], dtype=np.float32),
        # polygon_vertices: np.ndarray = np.array([
        #     [ 0.0, 0.0], [ 9.0, 1.0],
        #     [22.0, 2.0], [23.0, 8.0],
        #     [14.0, 9.0], [ 2.0, 6.0]
        # ], dtype=np.float32),
        # polygon_vertices: np.ndarray = np.array([
        #     [10.0, 16.0], [11.5, 12.0],
        #     [15.7, 11.9], [12.4, 9.2],
        #     [13.5, 5.1], [10.0, 7.5],
        #     [ 6.5, 5.1], [ 7.6, 9.2],
        #     [ 4.3, 11.9], [ 8.5, 12.0]
        # ], dtype=np.float32),
        num_agents: int = 6,
        r: float = 2.0,
        action_type=DISCRETE_ACT,
        max_agent_speed: Optional[float] = 3.0,
        max_agent_accel: Optional[float] = 3.0,
        max_steps: int = 500,
        **kwargs,
    ):
        self.grid_size = 256    # 计算覆盖率和重叠率是用到的分辨率，1/grid_size 是每个格栅的面积
        self.poly = Polygon(polygon_vertices)  # 仅供外部查看，多边形几何
        self.poly_verts = jnp.array(polygon_vertices)
        self.num_agents = num_agents
        self.r = r
        # 观察空间维度：
        local_obs_dim      = 2 + 2 + 2 + 1 + 1      # = 8
        nei_feat_dim       = 2                      # 单邻居特征 = 相对 (dx,dy)
        max_neighbors      = 9      # 槽位上限（缺省 9）
        concat_obs_dim     = local_obs_dim + nei_feat_dim * max_neighbors
        agents = [f"agent_{i}" for i in range(num_agents)]
        # == 直接用一根向量作为 Box ==
        obs_space = Box(-jnp.inf, jnp.inf, (concat_obs_dim,), dtype=jnp.float32)
        observation_spaces = {a: obs_space for a in agents}

        # 把这些尺寸存下来，get_obs 要用
        self.local_dim       = local_obs_dim
        self.nei_feat_dim    = nei_feat_dim
        self.max_neighbors   = max_neighbors
        self.concat_obs_dim  = concat_obs_dim
        super().__init__(
            num_agents=num_agents,
            agents=agents,
            num_landmarks=0,
            landmarks=[],
            action_type=action_type,
            observation_spaces=observation_spaces,
            dim_c=0,
            dim_p=2,
            colour=[AGENT_COLOUR]*num_agents,
            rad=jnp.full((num_agents,), r),
            collide=jnp.full((num_agents,), False),
            max_steps=max_steps,
            max_speed=jnp.full((num_agents,), max_agent_speed),
            accel=jnp.full((num_agents,), max_agent_accel),
            **kwargs,
        )

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: jax.random.PRNGKey) -> Tuple[Dict[str, jnp.ndarray], State]:
        """将智能体随机生成在多边形外部的同心圆周上"""
        key_agents = jax.random.split(key, self.num_agents)

        # 1. 计算多边形的质心
        centroid = jnp.mean(self.poly_verts, axis=0)

        # 2. 计算智能体生成圆周的半径
        #    首先计算多边形顶点到质心的最大距离
        dists_to_centroid = jax.vmap(lambda vert: jnp.linalg.norm(vert - centroid))(self.poly_verts)
        max_dist_to_centroid = jnp.max(dists_to_centroid)
        #    生成半径 = 最大顶点距离 + 智能体半径 + 一个小的偏移量（例如，智能体半径的一半）
        spawn_radius = max_dist_to_centroid + self.r + self.r * 0.5

        # 3. 为每个智能体生成随机角度
        # 4. 计算初始位置
        def get_agent_pos(k_agent):
            # 按照智能体数量在圆周上等分放置
            agent_idx = jnp.arange(self.num_agents)[jnp.where(key_agents == k_agent, size=1)[0][0]]
            angle = 2 * jnp.pi * agent_idx / self.num_agents
            pos_x = centroid[0] + spawn_radius * jnp.cos(angle)
            pos_y = centroid[1] + spawn_radius * jnp.sin(angle)
            return jnp.array([pos_x, pos_y])

        agent_p_pos = jax.vmap(get_agent_pos)(key_agents)
        
        # 由于 num_landmarks = 0, 我们不需要为 landmarks 生成位置
        # 如果有 landmarks, 需要像父类一样处理:
        # key_l = jax.random.split(key_agents[-1], 1)[0] # or some other key split logic
        # landmark_p_pos = jax.random.uniform(key_l, (self.num_landmarks, 2), minval=-1.0, maxval=+1.0)
        # p_pos = jnp.concatenate([agent_p_pos, landmark_p_pos])
        # 但在此特定环境中，num_landmarks 为 0，所以 p_pos 就是 agent_p_pos
        p_pos = agent_p_pos

        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)), # num_entities is num_agents here
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents,), False),
            step=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: State) -> Dict[str, jnp.ndarray]:
        @partial(jax.vmap, in_axes=(0,))
        def _obs(aidx):
            pos = state.p_pos[aidx]
            vel = state.p_vel[aidx]

            # === local 部分 ===
            I_i   = point_in_poly(pos, self.poly_verts)
            d_p   = dist_to_poly(pos, self.poly_verts)
            hat_d = closest_point_on_poly_relative(pos, self.poly_verts)  # 到边界的最近相对位移
            delta_i = jnp.where(I_i,
                                jnp.maximum(0.0, self.r - d_p),
                                self.r + d_p)

            local_vec = jnp.concatenate([pos, vel, hat_d,
                                        jnp.array([delta_i, I_i])])   # (8,)

            # === neighbor 相对位置 ===
            rel = state.p_pos - pos                                   # (N,2)
            
            # 创建一个 (N-1, 2) 的数组来存储其他智能体的相对位置
            # 使用条件赋值而不是动态切片
            def get_neighbor_positions():
                # 创建一个大小为 (N-1, 2) 的零数组
                neighbor_rel = jnp.zeros((self.num_agents - 1, 2))
                
                # 填充前半部分：索引 < aidx 的智能体
                def fill_before(i, arr):
                    # 如果 i < aidx，则填充 rel[i]，否则保持不变
                    updated_arr = jnp.where(
                        (i < aidx) & (i < self.num_agents - 1),
                        rel[i],
                        arr[i]
                    )
                    return arr.at[i].set(updated_arr)
                
                # 填充后半部分：索引 > aidx 的智能体
                def fill_after(i, arr):
                    # 如果 i >= aidx，则填充 rel[i+1]，否则保持不变
                    target_idx = i + 1  # 跳过 aidx
                    updated_arr = jnp.where(
                        (i >= aidx) & (target_idx < self.num_agents),
                        rel[target_idx],
                        arr[i]
                    )
                    return arr.at[i].set(updated_arr)
                
                # 使用 scan 来填充数组
                def scan_fill(carry, i):
                    arr = carry
                    # 先处理前半部分
                    arr = jnp.where(
                        i < aidx,
                        arr.at[i].set(rel[i]),
                        arr
                    )
                    # 再处理后半部分
                    arr = jnp.where(
                        i >= aidx,
                        arr.at[i].set(rel[i + 1]),
                        arr
                    )
                    return arr, None
                
                final_arr, _ = jax.lax.scan(scan_fill, neighbor_rel, jnp.arange(self.num_agents - 1))
                return final_arr
            
            rel_wo_self = get_neighbor_positions()  # (N-1, 2)

            # 固定槽位填充
            current_neighbors = rel_wo_self.shape[0]
            pad_len = self.max_neighbors - current_neighbors
            neigh = jnp.pad(rel_wo_self,
                            ((0, pad_len), (0, 0)),
                            mode='constant')                           # (max_neighbors,2)

            obs_flat = jnp.concatenate([local_vec, neigh.reshape(-1)])  # (concat_obs_dim,)
            return obs_flat

        # vmap 返回 (num_agents, obs_dim)；map 回字典
        obs_mat = _obs(jnp.arange(self.num_agents))
        return {a: obs_mat[i] for i, a in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=[0])
    def _compute_shaping(
        self,
        positions: jnp.ndarray,
        beta: float = 0.1,
    ) -> jnp.ndarray:
        """
        纯 JAX 计算各 agent 的势能塑形奖励 r_shape = -β * Φ_i。
        """
        delta0 = self.r / 2
        @jax.vmap
        def _U_h(pi):
            """边界势能 U_h"""
            inside = point_in_poly(pi, self.poly_verts)
            d_p = dist_to_poly(pi, self.poly_verts)
            delta = jnp.where(inside, jnp.maximum(0., self.r - d_p), self.r + d_p)
            return jnp.where(delta <= delta0, 0.0, 0.5 * (delta - delta0) ** 2)
        U_h_vec = _U_h(positions)
        # 互斥势能 U_I
        diffs = positions[:, None, :] - positions[None, :, :]
        dists = jnp.linalg.norm(diffs, axis=-1)
        thresh = 2 * self.r - delta0 * 2
        U_I = jnp.where(dists < thresh, 0.5 * (thresh - dists) ** 2, 0.0)
        sum_UI = jnp.sum(U_I, axis=1) - jnp.diag(U_I)
        # 总势能Φ = 0.75*U_h + 0.25*U_I
        Phi = 0.75 * U_h_vec + 0.25 * sum_UI
        return -beta * Phi

    @partial(jax.jit, static_argnums=[0])
    def rewards(self, state: State) -> Dict[str, jnp.ndarray]:
        """
        组合主奖励 (2*C - λ_u*U) 和势能塑形，并打包返回每个 agent 的 reward。
        全部函数均在 JIT 路径中执行。
        """
        positions = state.p_pos  # (num_agents, 2)
        # 势能塑形奖励
        shaping = self._compute_shaping(positions, beta=0.01)  # jnp array (num_agents,)
        
        # 覆盖率、重叠率和不公平度
        C, CAP, MCR, U, R_agents = compute_coverage_overlap(
            positions,
            self.poly_verts,
            self.r,
            grid_size=self.grid_size,
        )
        # C is scalar, R_agents is (num_agents,), R_global_orig is scalar, U is scalar

        # 主奖励: 2*C - λ_u*U (全局公平性惩罚)
        lambda_u = 1.0            # 公平性惩罚权重，可调
        main_agent_rewards = 2.0 * C - lambda_u * U  # 广播为 (num_agents,)

        # 里程碑奖励: 满足条件时给 B，利用 jnp.where 保持 JAX 兼容
        # Uses global C and U for the condition
        C_star, U_star, B = 0.95, 0.30, 5.0
        bonus_achieved = jnp.where((C >= C_star) & (U <= U_star), B, 0.0) # scalar

        # 距离过近的惩罚
        # 计算每个 agent 的距离惩罚
        distance_penalty = jnp.zeros((self.num_agents,))
        # 计算智能体间的距离矩阵
        diffs = positions[:, None, :] - positions[None, :, :]  # (N, N, 2)
        dists = jnp.linalg.norm(diffs, axis=-1)  # (N, N)

        # 创建掩码：距离小于 self.r / 2 且不是自身的情况
        penalty_mask = (dists < self.r / 2) & (dists > 0)  # 排除自身距离为0的情况

        # 检查每个agent是否有任何邻居距离过近（不重复计算惩罚）
        has_close_neighbor = jnp.any(penalty_mask, axis=1)  # (N,)

        # 每个agent的距离惩罚：只要有邻居过近就惩罚-1.0，多个邻居也不加重
        distance_penalty = jnp.where(has_close_neighbor, -1.0, 0.0)
        
        # 每个 agent 的最终 reward = main_reward + bonus_achieved_globally + shaping_i + distance_penalty
        final_rewards_per_agent = main_agent_rewards + bonus_achieved + shaping + distance_penalty

        return {a: final_rewards_per_agent[i] for i, a in enumerate(self.agents)}
    
    # @partial(jax.jit, static_argnums=[0])
    # def step_env(self, key: chex.PRNGKey, state: State, actions: dict) -> Tuple[Dict[str, chex.Array], State, Dict[str, chex.Array], Dict[str, bool], Dict]:
    #     """覆盖父类的 step_env 以实现自定义的 done 条件"""
    #     u, c_action = self.set_actions(actions) # c_action 对于 dim_c=0 是 (num_agents, 0)

    #     # 世界物理步骤，获取下一时刻的位置和速度
    #     key, key_w = jax.random.split(key)
    #     next_p_pos, next_p_vel = self._world_step(key_w, state, u)

    #     # 通信步骤 (对于此环境 c 通常是零向量)
    #     # 父类 SimpleMPE 的 _apply_comm_action 会处理 c_action
    #     # 如果 dim_c > 0, c_action 可能需要填充以匹配 self.dim_c
    #     # 但在此环境中 self.dim_c = 0, 所以 c_action 已经是 (num_agents, 0)
    #     # _apply_comm_action 也能正确处理 (num_agents, 0) 的输入
    #     key_c_noise = jax.random.split(key, self.num_agents)
    #     next_c = self._apply_comm_action(key_c_noise, c_action, self.c_noise, self.silent)


    #     # 计算覆盖率和重叠率以决定是否完成任务
    #     # We need C and R_global_orig for the done condition
    #     C, _, R_global_for_done = compute_coverage_overlap( # Modified call, R_agents is not used here
    #         next_p_pos, # 使用下一时刻的位置
    #         self.poly_verts,
    #         self.r,
    #         grid_size=self.grid_size,
    #     )

    #     # 完成条件判断
    #     C_star, R_star = 0.95, 0.15
    #     task_completed = (C >= C_star) & (R_global_for_done <= R_star) # Use R_global_for_done
    #     max_steps_reached = (state.step + 1) >= self.max_steps # 当前步数执行后是否达到或超过最大步数

    #     is_terminal_scalar = task_completed | max_steps_reached
    #     done_array = jnp.full((self.num_agents,), is_terminal_scalar)

    #     # 更新状态
    #     next_state = state.replace(
    #         p_pos=next_p_pos,
    #         p_vel=next_p_vel,
    #         c=next_c, # 更新通信状态
    #         done=done_array,
    #         step=state.step + 1,
    #     )

    #     # 计算奖励和观测
    #     reward = self.rewards(next_state)
    #     obs = self.get_obs(next_state)
        
    #     info = {} # 额外信息（如果需要）

    #     # 为 PettingZoo API 准备 dones 字典
    #     dones = {agent: done_array[i] for i, agent in enumerate(self.agents)}
    #     dones["__all__"] = jnp.all(done_array)

    #     return obs, next_state, reward, dones, info
