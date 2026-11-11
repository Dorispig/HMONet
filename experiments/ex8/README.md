# 自监督损失函数消融

基线采用正式 HMONet 的完整自监督目标：归一化 PDE 残差、归一化 Jacobian 屏障和归一化奇异值屏障。

| 名称 | 保留损失项 |
| --- | --- |
| `baseline` | PDE + Jacobian + SVD |
| `e1` | PDE |
| `e2` | PDE + Jacobian |
| `e3` | PDE + SVD |

## 命令

完整训练并评估：

```bash
python scripts/experiments/run.py ex8 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1
```

只训练两项组合：

```bash
python scripts/experiments/run.py ex8 e2 e3 --dataset air_t_ext_1_500_21_21 --seed 20261030 --tag v1 --train-only
```

只评估 PDE-only：

```bash
python scripts/experiments/run.py ex8 e1 --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1 --evaluate-only
```

## 质量图评估

二维 `--quality-plots` 可选项为：`all`、`max_angle`、`min_angle`、`jacobian`、`scaled_jacobian`、`skewness`、`aspect_ratio`、`orthogonality`、`concave_vertex_count`、`self_intersection`、`intersecting_cell`、`flipping_cell`、`degenerate_cell`。`all` 表示当前启用的全部指标；`cell_area` 已从统一评估中移除，不能作为质量图选项。

默认只保存完整质量汇总、各方法网格图和网格对比图。对已有损失函数变体生成指定质量图：

```bash
python scripts/experiments/run.py ex8 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1 --evaluate-only --quality-plots scaled_jacobian max_angle
```

使用 `--quality-plots all` 可输出当前启用的全部质量图及多方法质量对比图；指定图片不改变质量数值汇总。
