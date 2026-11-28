# 有向归一化 Jacobian 屏障权重消融

基线继承二维正式 HMONet 配置。训练目标为归一化 PDE 损失加上有向归一化 Jacobian 屏障损失。PDE 权重为 1，基线的 Jacobian 权重为 0。各变体只修改 Jacobian 权重。阈值沿用正式配置的 0.02，屏障在全部采样点上取平均，奇异值屏障权重为 0。

默认数据集为 `gsp_a_ext_1_500_41_41`，随机种子为 `20261030`，评估沿用二维正式配置中的振幅 `0.19 0.20`，默认评估规模为 `101×101`、`201×201`、`401×401` 个节点。训练轮数上限继承正式配置。此前的 300 轮离线测试与本套件分开保存。

| 短序号 | 完整名称 | Jacobian 权重 |
| --- | --- | ---: |
| `baseline` | `baseline` | 0 |
| `e1` | `e1_jac_w0_001` | 0.001 |
| `e2` | `e2_jac_w0_01` | 0.01 |
| `e3` | `e3_jac_w0_1` | 0.1 |
| `e4` | `e4_jac_w1` | 1 |
| `e5` | `e5_jac_w10` | 10 |
| `e6` | `e6_jac_w100` | 100 |
| `e7` | `e7_jac_w1000` | 1000 |
| `e8` | `e8_jac_w10000` | 10000 |

## 命令

训练并评估基线和全部变体：

```bash
python scripts/experiments/run.py ex9 baseline all --dataset gsp_a_ext_1_500_41_41 --tag v1
```

只运行指定变体的训练与评估：

```bash
python scripts/experiments/run.py ex9 e5 e6 --dataset gsp_a_ext_1_500_41_41 --tag v1
```

只训练：

```bash
python scripts/experiments/run.py ex9 e5 e6 --dataset gsp_a_ext_1_500_41_41 --tag v1 --train-only
```

只评估振幅 0.20 的 101×101 节点网格：

```bash
python scripts/experiments/run.py ex9 baseline e5 e6 --dataset gsp_a_ext_1_500_41_41 --tag v1 --evaluate-only --grids 101 --values 0.20
```

## 质量图

默认生成网格图及质量数值汇总。可用 `--quality-plots` 额外生成单方法及对比质量图。二维可选项为 `all`、`max_angle`、`min_angle`、`jacobian`、`scaled_jacobian`、`skewness`、`aspect_ratio`、`orthogonality`、`concave_vertex_count`、`self_intersection`、`intersecting_cell`、`flipping_cell`、`degenerate_cell`。

```bash
python scripts/experiments/run.py ex9 e6 --dataset gsp_a_ext_1_500_41_41 --tag v1 --evaluate-only --grids 101 --values 0.20 --quality-plots scaled_jacobian flipping_cell
```
