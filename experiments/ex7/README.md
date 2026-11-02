# ex7：Jacobian与非归一化PDE损失消融

本实验使用精简HMONet，各组统一开启主干网络的位置相关几何系数模块和曲边单元修正，只消融Jacobian与PDE损失。仅曲边修正结果由套件基线只读提供，实验运行以下三组：

1. `jacobian_only`：曲边修正 + Jacobian损失；
2. `pde_only`：曲边修正 + 非归一化PDE损失；
3. `jacobian_pde`：曲边修正 + Jacobian损失 + 非归一化PDE损失。

trunk模块改变了模型结构，因此ex7不复用ex4基线。基线只能用下列独立命令显式训练；消融命令不会触发它：

```powershell
python scripts/experiments/run_baseline_2d.py --experiment-set ex7 --dataset air_t_ext_0_1000_41_20 --seeds 20260802 20260917 20261031 --run-tag paper_v1
```

```powershell
python scripts/experiments/run_ablations_2d.py --experiment-set ex7 --ablations pde_jacobian --dataset air_t_ext_0_1000_41_20 --seeds 20260802 20260917 20261031 --run-tag paper_v1 --continue-on-error
```

如需只运行某些变体，请直接调用单实验入口：

```powershell
python scripts/experiments/run_ablation_2d.py --experiment-set ex7 --ablation pde_jacobian --dataset air_t_ext_0_1000_41_20 --variants pde_only jacobian_pde --seeds 20260802 20260917 20261031 --run-tag paper_v1 --continue-on-error
```

非归一化PDE损失采用椭圆网格生成方程的有限差分残差，默认独立采样128个内部点，权重为`1e-4`。
