# 坐标增强阶数消融

基线采用新的 power-Fourier 坐标增强，最高幂次 `P=2`；变体依次测试 `P=1,3,4,5`。

```bash
python scripts/experiments/run.py ex7 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1
```
