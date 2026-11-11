# HMONet 网络架构消融

基线使用正式 HMONet 的 CCAP 映射、幂次--Fourier 坐标增强和归一化复合物理目标。每项变体保持相同的输入、输出、硬边界和评估流程。

| 名称 | 相对基线的修改 |
| --- | --- |
| `baseline` | 正式 HMONet（作为所有架构变体的共同对照） |
| `e1` | 分支和主干改为门控 ModifiedMLP |
| `e2` | 增加查询点到边界的交叉注意力 |
| `e3` | 增加边界层专用解码器 |
| `e4` | 解耦原始坐标与映射权重路径 |
| `e5` | Query--Geometry Gated HMONet |
| `e6` | ModifiedMLP + 交叉注意力 |
| `e7` | 再增加边界层解码器 |
| `e8` | 再增加权重解耦 |
| `e9` | 约 20 万参数 MeshONet |
| `e10` | 约 20 万参数 PI-MeshONet |
| `e11` | 将归一化 PDE 替换为原始 PDE 残差 |

## 命令

完整训练并评估：

```bash
python scripts/experiments/run.py ex3 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1
```

只训练交叉注意力与边界解码器：

```bash
python scripts/experiments/run.py ex3 e2 e3 --dataset air_t_ext_1_500_21_21 --seed 20261030 --tag v1 --train-only
```

只评估 QG-HMONet：

```bash
python scripts/experiments/run.py ex3 e5 --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1 --evaluate-only
```

QG-HMONet不显式计算曲率、法向或SDF；它使用多尺度边界编码、查询—边界交叉注意力和门控融合，同时保留精确硬边界映射。

## 质量图评估

二维 `--quality-plots` 可选项为：`all`、`max_angle`、`min_angle`、`jacobian`、`scaled_jacobian`、`skewness`、`aspect_ratio`、`orthogonality`、`concave_vertex_count`、`self_intersection`、`intersecting_cell`、`flipping_cell`、`degenerate_cell`。`all` 表示当前启用的全部指标；`cell_area` 已从统一评估中移除，不能作为质量图选项。

默认仅保存完整质量汇总、各方法网格图和网格对比图。为已有架构模型补充指定质量图：

```bash
python scripts/experiments/run.py ex3 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1 --evaluate-only --quality-plots scaled_jacobian max_angle
```

将末尾替换为 `--quality-plots all` 可生成当前启用的全部质量图和多方法质量对比图；数值汇总始终完整保留。

