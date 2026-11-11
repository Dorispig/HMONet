# 物理采样策略消融

基线继承正式二维 HMONet 配置，即 200 个全局均匀点与 200 个曲率--边界距离自适应点。所有变体只改变每批 400 个物理中心的组成。

| 名称 | 内容 |
| --- | --- |
| `baseline` | 200 均匀 + 200 自适应（正式默认） |
| `e1_1` | 400 个均匀点 |
| `e1_2` | 300 均匀 + 100 自适应 |
| `e1_3` | 100 均匀 + 300 自适应 |
| `e1_4` | 400 个自适应点 |

## 命令

完整训练并评估：

```bash
python scripts/experiments/run.py ex2 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1
```

只训练指定策略：

```bash
python scripts/experiments/run.py ex2 e1_1 e1_4 --dataset air_t_ext_1_500_21_21 --seed 20261030 --tag v1 --train-only
```

只评估指定策略：

```bash
python scripts/experiments/run.py ex2 e1_1 e1_4 --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1 --evaluate-only
```

## 质量图评估

二维 `--quality-plots` 可选项为：`all`、`max_angle`、`min_angle`、`jacobian`、`scaled_jacobian`、`skewness`、`aspect_ratio`、`orthogonality`、`concave_vertex_count`、`self_intersection`、`intersecting_cell`、`flipping_cell`、`degenerate_cell`。`all` 表示当前启用的全部指标；`cell_area` 已从统一评估中移除，不能作为质量图选项。

默认仅保留完整质量汇总、各方法网格图和网格对比图。对已有模型只评估并生成指定质量图：

```bash
python scripts/experiments/run.py ex2 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1 --evaluate-only --quality-plots scaled_jacobian max_angle
```

将末尾改为 `--quality-plots all` 可生成当前启用的全部质量图及相应多方法对比图；不影响完整数值质量汇总。

