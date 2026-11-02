# ex8：HMONet 多阶段训练策略消融

基线为 `baseline/experiment.yaml` 中的当前监督式 HMONet。所有变体各使用三个随机种子训练，并沿用同一套评估与汇总流程。

| 变体 | 阶段 1 | 阶段 2 |
| --- | --- | --- |
| `e1_mse_pde` | 无第一阶段 | 固定网格随机 400 点 MSE + 随机 400 点 `1e-4 L_PDE` |
| `e2_nojac_then_mse_pde` | 无 Jacobian 的当前随机点 HMONet，直到成功收敛 | 固定网格随机 400 点 MSE + 随机 400 点 `1e-4 L_PDE` |
| `e3_jac_then_mse_pde` | 含 Jacobian 损失的当前随机点 HMONet，直到成功收敛 | 固定网格随机 400 点 MSE + 随机 400 点 `1e-4 L_PDE` |
| `e4_nojac_then_pde` | 当前不含 Jacobian 损失的监督 HMONet，直到成功收敛 | 仅随机 PDE 点的 `L_PDE` |
| `e5_jac_then_pde` | 当前含 Jacobian 损失的监督 HMONet，直到成功收敛 | 仅随机 PDE 点的 `L_PDE` |

实验2--5在阶段1结束时保存 `phase1_best.pt`。进入阶段2后保留阶段1的最佳记录，
并继续使用阶段1的 Adam 状态、当前学习率及全局学习率调度，不重新预热或放大学习率。
常规 `best.pt` 在两个阶段的全部迭代中统一选择全程最佳模型。评估分别读取阶段1
最佳模型和全程最佳模型，汇总名称使用 `__phase1` 和 `__global` 区分。

所有 ex8 运行的总上限为 100000 次；阶段切换后仍使用同一个总上限。固定 MSE 点和 PDE 点均在每批重新随机，且同一批样本共享同一组逻辑点。

如确实缺少参照基线，可用独立命令显式创建；后续消融命令只读检查基线指纹，不会训练或评估基线：

```powershell
python scripts/experiments/run_baseline_2d.py --experiment-set ex8 --dataset air_t_ext_0_1000_41_20 --seeds 20261030 --run-tag physics_v1 --evaluate
```

再运行五个阶段训练变体及其汇总评估：

```powershell
python scripts/experiments/run_ablations_2d.py --experiment-set ex8 --ablations physics_training_strategy --dataset air_t_ext_0_1000_41_20 --seeds 20261030 --run-tag physics_v1 --continue-on-error
```

只运行五个实验时使用精简名称：

```powershell
python scripts/experiments/run_ablation_2d.py --experiment-set ex8 --ablation physics_training_strategy --variants e1_mse_pde e2_nojac_then_mse_pde e3_jac_then_mse_pde e4_nojac_then_pde e5_jac_then_pde --dataset run_ext_0_1000_61_21 --seeds 20261030 --run-tag physics_v1 --evaluation-grid-shapes 101x101 201x201 401x401 --continue-on-error
```

对于“监督后继续训练”变体，只有第一阶段以正常成功收敛结束时才进入第二阶段；如果第一阶段触发失败平台或达到阶段上限，训练会结束并标记为该阶段未成功收敛。
