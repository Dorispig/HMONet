# Jacobian/SVD 屏障超参数消融

基线采用正式 HMONet 的 `w_J=0.01,\ \tau_J=0.02,\ w_\sigma=0.05,\ \tau_\sigma=0.05`；每个变体只修改一个权重或阈值。

| 组别 | 变体 | 内容 |
| --- | --- | --- |
| 正式基线 | `baseline` | $w_J=0.01,\ \tau_J=0.02,\ w_\sigma=0.05,\ \tau_\sigma=0.05$ |
| Jacobian 权重 | `e1_1`--`e1_6` | `100, 10, 1, 0.1, 0.001, 0.0001` |
| Jacobian 阈值 | `e2_1`--`e2_6` | `0.001, 0.005, 0.01, 0.05, 0.1, 0.2` |
| SVD 权重 | `e3_1`--`e3_6` | `100, 10, 1, 0.1, 0.01, 0.001` |
| SVD 阈值 | `e4_1`--`e4_5` | `0.001, 0.005, 0.01, 0.1, 0.2` |

`0.05` 是 SVD 阈值基线，故不重复建立对应变体。

## 命令

完整训练并评估：

```bash
python scripts/experiments/run.py ex5 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1
```

只训练 Jacobian 权重组：

```bash
python scripts/experiments/run.py ex5 e1_1 e1_2 e1_3 e1_4 e1_5 e1_6 --dataset air_t_ext_1_500_21_21 --seed 20261030 --tag v1 --train-only
```

只评估 SVD 阈值组：

```bash
python scripts/experiments/run.py ex5 e4_1 e4_2 e4_3 e4_4 e4_5 --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1 --evaluate-only
```
