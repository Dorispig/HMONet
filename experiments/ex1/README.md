# 硬边界约束消融

本套件比较三种精确硬边界映射。`baseline` 直接继承当前正式二维配置；两项变体除硬边界公式外不覆盖任何训练、采样、模型或收敛设置。默认数据集为 `air_t_ext_1_500_21_21`。

| 名称 | 内容 |
| --- | --- |
| `baseline` | CCAP角点校正仿射投影 |
| `rational_affine` | RAP有理仿射投影 |
| `coons_residual` | Coons基函数加网络残差 |

训练并评估三种硬约束：

```bash
python scripts/experiments/run.py ex1 baseline rational_affine coons_residual --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1
```

只复用已有结果进行评估：

```bash
python scripts/experiments/run.py ex1 baseline rational_affine coons_residual --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1 --evaluate-only
```

完整数值汇总见[硬边界约束公式对比](../reports/hard_boundary_mapping_comparison.md)。

