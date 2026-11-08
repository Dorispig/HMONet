# 自监督损失函数消融

基线使用完整目标：归一化 PDE 残差、归一化 Jacobian 屏障和归一化奇异值屏障。三个变体分别测试：

1. 仅归一化 PDE；
2. 归一化 PDE + 归一化 Jacobian；
3. 归一化 PDE + 归一化 SVD。

```bash
python scripts/experiments/run.py ex8 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1
```
