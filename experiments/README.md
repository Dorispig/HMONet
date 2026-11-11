# 消融实验套件

活动消融实验已经按当前纯自监督 HMONet 的研究问题重新编号。旧监督式、曲边目标插值和分阶段微调实验已清理，其结论保存在 [reports](reports/) 中。

| 套件 | 研究问题 | 主要实验 |
| --- | --- | --- |
| `ex1` | 精确硬边界约束 | CCAP基线、RAP、Coons残差 |
| `ex2` | 采样策略与对比模型 | 均匀/自适应/边界层采样、MeshONet、PI-MeshONet |
| `ex3` | 网络架构 | ModifiedMLP、交叉注意力、边界层解码器、解耦权重、QG-HMONet及组合 |
| `ex4` | PDE求导方式 | 安全中心差分、全开域差分、自动微分、边界单边差分 |
| `ex5` | 屏障权重与阈值 | 归一化Jacobian/SVD的权重和阈值扫描 |
| `ex6` | 有限差分步长 | PDE与Jacobian共用差分步长扫描 |
| `ex7` | 坐标增强阶数 | power-Fourier最高幂次 `P=1,2,3,4,5` |
| `ex8` | 损失项组成 | 归一化PDE、PDE+Jacobian、PDE+SVD、完整目标 |

## 统一命令

```bash
python scripts/experiments/run.py <suite> [baseline] <experiment|all> [options]
```

`all`只运行套件内变体，不包含基线；只有显式写出`baseline`才会运行基线。常用参数：

- `--dataset NAME`：覆盖数据集；
- `--seed S...`：一个或多个随机种子；
- `--grids N...`：评估网格规模；
- `--values V...`：评估参数；
- `--tag TAG`：输出标签（仅限字母、数字、`.`、`_`、`-`，最长 24 字符）；
- `--gpus 0 1 2 3`：动态多GPU任务队列；
- `--train-only`或`--evaluate-only`：只执行一个阶段；
- `--force`：替换所选训练与评估；
- `--no-plots`：不生成任何新评估图片；
- `--quality-plots all`：额外生成全部质量指标图与对应比较图；
- `--quality-plots scaled_jacobian max_angle`：只生成指定指标的单方法图和多方法对比图。默认不生成质量图片，完整数值汇总仍会保留。

完整评估结果统一写入：

```text
experiments/<exN>/eval/<dataset>/seed_<seed>/<run-tag>/<experiment>/<grid>/<value>/
```

其中图片采用紧凑子目录 `p/`：默认仅保留各方法的原始/后处理网格图与 `p/c/<methods>/mesh_comparison.png`。显式传入 `--quality-plots` 后，质量图才会写入 `p/c/<methods>/q/` 和 `p/<method>/r|p/q/`。方法和完整指标名称始终保留在同层 `results.json` 中。

多GPU日志写入：

```text
experiments/<exN>/queue_logs/<dataset>/<experiment>/seed_<seed>/<run-tag>/
```

每个套件的实验名和命令见对应目录README；旧实验的研究结论见：

- [旧监督式、分阶段训练与容量数据量实验汇总](reports/legacy_supervised_and_staged_ablation_summary.md)
- [自监督物理损失探索与失败模式汇总](reports/self_supervised_loss_exploration_summary.md)
- [硬边界约束公式对比](reports/hard_boundary_mapping_comparison.md)
- [旧架构变体质量对比](reports/legacy_architecture_variants_quality_comparison.md)

