# PDE/Jacobian 有限差分步长消融

基线中 PDE 与归一化 Jacobian 的有限差分步长均为 `0.01`；每个变体同步修改两者。

| 名称 | 步长 |
| --- | --- |
| `baseline` | `0.01` |
| `e1_1`--`e1_5` | `0.0005, 0.001, 0.005, 0.05, 0.1` |
| `e1_6` | `0.008` |
| `e1_7` | `0.02` |
| `e1_8` | `0.03` |

## 命令

完整训练并评估：

```bash
python scripts/experiments/run.py ex6 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1
```

只训练新增的中间步长：

```bash
python scripts/experiments/run.py ex6 e1_6 e1_7 e1_8 --dataset air_t_ext_1_500_21_21 --seed 20261030 --tag v1 --train-only
```

只评估新增的中间步长：

```bash
python scripts/experiments/run.py ex6 e1_6 e1_7 e1_8 --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1 --evaluate-only
```
