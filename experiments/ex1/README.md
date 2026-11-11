# 硬边界约束消融

基线继承正式二维 HMONet 配置；所有变体只修改硬边界公式。默认数据集为 `air_t_ext_1_500_21_21`，默认评估参数为 `0.62 0.65`。

| 名称 | 内容 |
| --- | --- |
| `baseline` | CCAP 角点校正仿射投影（正式默认） |
| `e1_1` | RAP 有理仿射投影 |
| `e1_2` | Coons 基函数与网络残差 |

## 命令

完整训练并评估：

```bash
python scripts/experiments/run.py ex1 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1
```

只训练指定变体：

```bash
python scripts/experiments/run.py ex1 e1_1 e1_2 --dataset air_t_ext_1_500_21_21 --seed 20261030 --tag v1 --train-only
```

只评估指定变体：

```bash
python scripts/experiments/run.py ex1 e1_1 e1_2 --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1 --evaluate-only
```

`baseline all` 表示基线加全部变体；只写 `all` 表示仅全部变体，不重复运行基线。

## 质量图评估

二维 `--quality-plots` 可选项为：`all`、`max_angle`、`min_angle`、`jacobian`、`scaled_jacobian`、`skewness`、`aspect_ratio`、`orthogonality`、`concave_vertex_count`、`self_intersection`、`intersecting_cell`、`flipping_cell`、`degenerate_cell`。`all` 表示当前启用的全部指标；`cell_area` 已从统一评估中移除，不能作为质量图选项。

默认评估仅保存完整质量汇总、各方法网格图和网格对比图，不生成质量热图。对已有模型仅补充指定质量图：

```bash
python scripts/experiments/run.py ex1 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1 --evaluate-only --quality-plots scaled_jacobian max_angle
```

生成当前启用的全部质量图及对应多方法对比图：将末尾替换为 `--quality-plots all`。质量图选择不影响所有质量指标的数值汇总。

