# 坐标增强消融

基线采用正式 HMONet 的 power--Fourier 坐标增强，最高幂次为 `P=2`，
二维主干输入为 12 维。实验分为“新增强的阶数”和“新旧增强公式”两组；
除坐标增强外，模型、硬边界、损失和采样设置均继承基线。

| 组别 | 名称 | 坐标增强 | 主干输入维度 | 比较目的 |
| --- | --- | --- | ---: | --- |
| 阶数 | `baseline` | power--Fourier，`P=2` | 12 | 正式配置 |
| 阶数 | `e1_1` | power--Fourier，`P=1` | 6 | 低阶端点 |
| 阶数 | `e1_2` | power--Fourier，`P=3` | 18 | 更高阶 |
| 阶数 | `e1_3` | power--Fourier，`P=4` | 24 | 更高阶 |
| 阶数 | `e1_4` | power--Fourier，`P=5` | 30 | 更高阶 |
| 公式 | `e_2_1` | legacy | 12 | 与基线等输入维度，隔离增强公式 |
| 公式 | `e_2_2` | legacy | 18 | 与 power--Fourier，`P=3` 等输入维度 |

## 命令

完整训练并评估：

```bash
python scripts/experiments/run.py ex7 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1
```

只训练低阶与高阶端点：

```bash
python scripts/experiments/run.py ex7 e1_1 e1_4 --dataset air_t_ext_1_500_21_21 --seed 20261030 --tag v1 --train-only
```

只评估指定阶数：

```bash
python scripts/experiments/run.py ex7 e1_2 e1_3 --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1 --evaluate-only
```

只比较新旧坐标增强公式：

```bash
python scripts/experiments/run.py ex7 baseline e_2_1 e_2_2 --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1
```

## 质量图评估

二维 `--quality-plots` 可选项为：`all`、`max_angle`、`min_angle`、`jacobian`、`scaled_jacobian`、`skewness`、`aspect_ratio`、`orthogonality`、`concave_vertex_count`、`self_intersection`、`intersecting_cell`、`flipping_cell`、`degenerate_cell`。`all` 表示当前启用的全部指标；`cell_area` 已从统一评估中移除，不能作为质量图选项。

默认只保存完整质量汇总、各方法网格图和网格对比图。对已有坐标增强变体补充指定质量图：

```bash
python scripts/experiments/run.py ex7 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1 --evaluate-only --quality-plots scaled_jacobian max_angle
```

将末尾替换为 `--quality-plots all` 可生成当前启用的全部质量图及多方法质量对比图；数值质量汇总仍完整保存。
