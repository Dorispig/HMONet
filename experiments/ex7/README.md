# ex7：Jacobian与非归一化PDE损失消融

本实验使用精简HMONet，四组统一开启主干网络的位置相关几何系数模块和曲边单元修正，只消融Jacobian与PDE损失：

1. `baseline`：仅曲边修正；
2. `jacobian_only`：曲边修正 + Jacobian损失；
3. `pde_only`：曲边修正 + 非归一化PDE损失；
4. `jacobian_pde`：曲边修正 + Jacobian损失 + 非归一化PDE损失。

trunk模块改变了模型结构，因此ex7不复用ex4模型，四组均需在ex7中重新训练。先训练共享基线，再运行完整消融：

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
