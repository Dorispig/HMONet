# 物理采样策略消融

本套件的基线直接继承当前正式二维 HMONet 配置（CCAP、归一化 PDE/Jacobian/SVD 损失及其余正式超参数）。每个变体仅改变 400 个物理中心的组成。默认数据集为 `air_t_ext_1_500_21_21`。

- `uniform_400`：400个全局均匀点；
- `adaptive_400`：400个曲率—边界距离自适应点；
- `global300_adaptive100`：300个全局均匀点 + 100个曲率—距离自适应点；
- `global100_adaptive300`：100个全局均匀点 + 300个曲率—距离自适应点；

```bash
python scripts/experiments/run.py ex2 baseline uniform_400 global300_adaptive100 global100_adaptive300 adaptive_400 --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1
```

四卡动态运行时追加`--gpus 0 1 2 3`。如需测试其他数据集，请显式传入对应的 `--dataset` 和 `--values`。

