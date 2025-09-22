# Neighbor-Embedding MAPPO (PyTorch)

本仓库将原始的 `mappo_rnn_neighbor_embed_mpe.py`（JAX 版本）重写为 **单文件的 PyTorch** 实现，
遵循 FreeRL 示例中“一个脚本包含环境、模型与训练循环”的约定。新的脚本名为
`freerl_neighbor_embed_mappo.py`，同时保留了覆盖任务的关键特性：

- 多智能体多边形覆盖环境，支持邻居信息拼接、覆盖率/重叠率/公平性度量以及简单的奖励设计；
- 邻居嵌入版 MAPPO：Actor 利用局部观测 + 邻居特征，经嵌入与 GRU 产生策略；Critic 接收展平后的世界状态；
- GAE + PPO 的训练流程，支持多环境并行采样、学习率退火以及梯度裁剪。

## 运行方式

```bash
python freerl_neighbor_embed_mappo.py --config mappo_homogenous_rnn_mpe_embed.yaml
```

如不指定 `--config`，脚本将使用内置的默认配置（见 `NeighborEmbedConfig` 数据类）。配置项覆盖环境
尺寸、时间步数、网络结构、PPO 超参数以及随机种子等内容。

## 代码结构

- `SimpleCoverageEnv`：纯 NumPy 实现的多边形覆盖环境，内部提供覆盖率统计、邻居特征拼接和自动重置。
- `CoverageVecEnv`：简易的同步向量环境封装器，用于同时运行多个环境实例。
- `NeighborEmbedMAPPO`：封装采样、优势计算与参数更新的训练器；Actor/Critic 采用 PyTorch GRUCell。

## TODO

- [ ] 对覆盖环境和奖励函数进行更精细的验证，确保与 JAX 版本在数值上保持一致。
- [ ] 根据需要补充 WandB/Checkpoint 等工程化组件。
- [ ] 实现基于 GPU 的批量可视化或评估脚本。
