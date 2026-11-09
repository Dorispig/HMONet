# 坐标增强阶数消融

基线采用正式 HMONet 的 power--Fourier 坐标增强，最高幂次为 `P=2`。

| 名称 | 最高幂次 |
| --- | --- |
| `baseline` | `P=2` |
| `e1_1` | `P=1` |
| `e1_2` | `P=3` |
| `e1_3` | `P=4` |
| `e1_4` | `P=5` |

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
