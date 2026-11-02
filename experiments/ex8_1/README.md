# ex8_1：Coons 残差硬约束下的多阶段训练策略

本实验集直接引用 `ex8/baseline` 进行参数兼容性检查，不创建、不训练、不评估也不修改自己的基线。五个实验与 ex8 的训练策略逐项对应，唯一统一变化是：

```text
hmonet_mapping_mode = coons_residual
```

| 变体 | 阶段 1 | 阶段 2 |
| --- | --- | --- |
| `e1_mse_pde` | 无第一阶段 | 固定网格随机 400 点 MSE + 随机 400 点 `1e-4 L_PDE` |
| `e2_nojac_then_mse_pde` | 无 Jacobian 的监督训练 | 固定网格随机 400 点 MSE + 随机 400 点 `1e-4 L_PDE` |
| `e3_jac_then_mse_pde` | 含 Jacobian 的监督训练 | 固定网格随机 400 点 MSE + 随机 400 点 `1e-4 L_PDE` |
| `e4_nojac_then_pde` | 无 Jacobian 的监督训练 | 仅 `L_PDE` |
| `e5_jac_then_pde` | 含 Jacobian 的监督训练 | 仅 `L_PDE` |

实验2--5在阶段切换时保留 Adam 状态、当前学习率和全局学习率调度；只重置第二阶段的收敛统计窗口。

训练并评估五个实验（一个随机种子）：

```powershell
python scripts/experiments/run_ablations_2d.py --experiment-set ex8_1 --ablations physics_training_strategy --dataset run_ext_0_1000_61_21 --seeds 20261030 --run-tag physics_v1 --evaluation-grid-shapes 101x101 201x201 401x401 --continue-on-error
```

若 ex8 中缺少相同数据集、随机种子和指纹的完整基线，脚本会在五个变体完成可执行训练后阻止基线对比汇总，并提示先训练 ex8 基线；它不会在 ex8_1 中补训或补评估基线。`--force` 也只作用于五个 Coons 变体，不覆盖 ex8 基线。
