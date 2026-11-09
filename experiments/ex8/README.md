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
