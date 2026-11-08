# PDE/Jacobian有限差分步长消融

基线中 PDE 和 Jacobian 有限差分步长均为 `0.01`。五个变体将二者同步改为 `0.0005, 0.001, 0.005, 0.05, 0.1`。

```bash
python scripts/experiments/run.py ex6 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1
```
