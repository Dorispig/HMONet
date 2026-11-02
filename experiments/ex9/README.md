# ex9：随机点 MSE / PDE / Jacobian 损失组合

二维 `run_ext_0_1000_61_21`，采用 `corner_corrected` 硬边界约束。所有模型从随机初始化开始训练，总上限为 50,000 epoch。通常每批重新抽取 400 个均匀随机内部点；随机点 MSE 使用曲边单元修正插值得到目标，所有点权重为 1。PDE 与 Jacobian 共用另一组 400 个随机中心点及有限差分模板。实验 6 第二阶段的 MSE 改为从数据集固定内部网格节点中随机选取 400 点。

| 组别 | 训练损失 | 最佳模型依据 |
| --- | --- | --- |
| `e1_mse_pde` | `L_MSE + 1e-4 L_PDE` | 固定网格 MSE |
| `e2_pde_jac` | `L_PDE + 1e-4 L_J` | 训练损失 |
| `e3_jac_pde` | `L_J + 1e-4 L_PDE` | 训练损失 |
| `e4_mse_pde_jac` | `L_MSE + 1e-4 L_PDE + 1e-4 L_J` | 固定网格 MSE |
| `e5_staged_mse_pde_jac` | 前 5000 轮 `L_MSE + 1e-4 L_J`，之后 `L_MSE + 1e-4 L_PDE + 1e-4 L_J` | 固定网格 MSE |
| `e6_pde_fixed_mse` | 前 5000 轮同实验 5；从第一阶段最佳模型以 `L_MSE,fixed + λ_PDE L_PDE + 1e-4 L_J + L_keep` 微调 | 固定网格 MSE，并检查网格合法性 |
| `e7_pde_winslow_jac` | `0.001 L_PDE + L_W,reg + 0.01 L_J,barrier`，不使用内部网格标签 | 训练损失 |
| `e8_pde_only` | `L_PDE`，不使用内部网格标签或其他正则项 | 训练损失 |
| `e9_norm_pde_jac_svd` | 尺度归一化 `L_PDE + 0.01 L_J,norm + 0.05 L_sigma,min` | 训练损失 |
| `e9_1` | e9损失；200个全局点、100个曲率自适应点、100个距边界1--4个有限差分步长的点；有限差分求导 | 训练损失与最差网格质量 |
| `e9_2` | 与e9_1相同的400点组成；自动微分求导，近边界距离改为0--0.0004 | 训练损失与最差网格质量 |
| `e9_3` | 与e9_1完全相同，但硬边界约束由`corner_corrected`改为`current` | 硬约束形式对高曲率区域和最差网格质量的影响 |

`L_PDE` 是非归一化椭圆残差；`L_J` 的阈值为 `1e-4`。不同权重的两组纯物理实验用于检验损失主导项对网格退化与相交的影响。纯物理组仍记录固定网格 MSE 供分析，但不使用它挑选检查点。

参照基线不属于上述实验，也不会由仅选择实验的命令自动运行。仅在明确需要建立基线时单独执行：

```powershell
python scripts/experiments/run_baseline_2d.py --experiment-set ex9 --dataset run_ext_0_1000_61_21 --seeds 20261030 --run-tag physics_v1 --evaluate
```

各实验分别在独立目录中配置；混合采样实验位于 `e9_1/`、`e9_2/` 和 `e9_3/`。建议每条命令只选择当前需要的实验；无论选择一个还是多个，均不会自动运行基线。

例如只运行实验 1：

```powershell
python scripts/experiments/run_ablations_2d.py --experiment-set ex9 --ablations e1_mse_pde --dataset run_ext_0_1000_61_21 --seeds 20261030 --run-tag physics_v1
```

默认评估 `101x101`、`201x201`、`401x401`。命令行中的 `--seeds` 和 `--evaluation-grid-shapes` 可覆盖默认值。重跑时，指纹匹配的完整训练和评估结果会复用；`--dry-run` 可先查看计划，`--force` 才重新计算。

实验 5 单独运行：

```powershell
python scripts/experiments/run_ablations_2d.py --experiment-set ex9 --ablations e5_staged_mse_pde_jac --dataset run_ext_0_1000_61_21 --seeds 20261030 --run-tag physics_v1 --evaluation-grid-shapes 101x101 201x201 401x401
```

实验 5 的初始学习率为 `0.0004`，阶段切换时保留优化器及学习率进度；其余实验的配置不变。

实验 6 与实验 5 的第一阶段相同。阶段切换时加载第一阶段最佳模型，保存冻结副本，第二阶段以固定内部节点 MSE 为主项，保留 `1e-4` Jacobian 损失，并用冻结模型的随机内部点输出构成保持损失。非归一化 PDE 的权重根据切换后首批 MSE/PDE 梯度范数校准，使目标 PDE 梯度为 MSE 梯度的 10%（权重上限 `1e-6`），随后在 1000 次迭代内线性增加。第二阶段学习率为切换时的 10%，重新建立 Adam 动量，不重置全局学习率调度。每批 MSE 和 PDE 各取 400 点。第二阶段训练期间不计算网格质量；训练结束后仅对最佳模型与第一阶段模型进行一次质量比较，若退化则使用第一阶段最佳模型。完整训练指纹由修改后的实验配置决定，已有实验 6 结果不会被误当成新的微调结果。

实验 6 单独运行：

```powershell
python scripts/experiments/run_ablations_2d.py --experiment-set ex9 --ablations e6_pde_fixed_mse --dataset run_ext_0_1000_61_21 --seeds 20261030 --run-tag physics_v1 --evaluation-grid-shapes 101x101 201x201 401x401
```

实验 7 的三项损失在同一批 400 个随机内部中心点上计算。PDE 使用 `pde_fd_step`（null 时为训练网格两轴间距）；Winslow 和其 Jacobian 屏障使用 `jacobian_fd_step`（当前基线配置为 `1e-4`，两轴相同）。两套查询合并为一次模型前向。`L_W,reg` 使用 `(||X_xi||²+||X_eta||²)/chi(J,epsilon)`，其中 `chi=(J+sqrt(J²+epsilon²))/2`、`epsilon=0.01`。`L_J,barrier=mean(max(0,1-J/0.01)²)` 是无量纲的局部面积下限惩罚；它与其他实验使用的未归一化 `L_J` 不同。Winslow 项改善局部形状，Jacobian 屏障避免其在接近各向同性塌缩时失效。三个系数 `0.001`、`1`、`0.01` 是当前实验值，不表示最优权重。训练的梯度只来自边界输入与模型预测；每个既定测量周期结束后额外计算并输出完整固定训练网格内部点的 MSE，作为诊断记录，但仍按训练损失收敛与选模，该 MSE 不参与反向传播。

实验 7 单独训练并评估：

```powershell
python scripts/experiments/run_ablations_2d.py --experiment-set ex9 --ablations e7_pde_winslow_jac --dataset run_ext_0_1000_61_21 --seeds 20261030 --run-tag physics_v1 --evaluation-grid-shapes 101x101 201x201 401x401
```

实验 8 仅使用随机内部点上的非归一化 PDE 残差，不使用随机点 MSE、Winslow 能量或任何 Jacobian 损失；固定训练网格 MSE 只作为诊断记录，不参与反向传播、收敛判断或最佳模型选择。

实验 8 单独训练并评估：

```powershell
python scripts/experiments/run_ablations_2d.py --experiment-set ex9 --ablations e8_pde_only --dataset run_ext_0_1000_61_21 --seeds 20261030 --run-tag physics_v1 --evaluation-grid-shapes 101x101 201x201 401x401
```

实验 9 使用尺度不变的椭圆 PDE 残差。Jacobian 与最小奇异值均按几何包围盒尺度归一化，并且只对违反下界的采样点求平均；达到 `J_norm >= 0.02` 和 `sigma_min,norm >= 0.05` 后，对应正则项自然严格变为零，不需要额外的权重衰减日程。

实验 9 单独训练并评估：

```powershell
python scripts/experiments/run_ablations_2d.py --experiment-set ex9 --ablations e9_norm_pde_jac_svd --dataset run_ext_0_1000_61_21 --seeds 20261030 --run-tag physics_v1 --evaluation-grid-shapes 101x101 201x201 401x401
```

混合采样的有限差分/自动微分实验可分别运行，也可在一条命令中连续运行：

```powershell
python scripts/experiments/run_ablations_2d.py --experiment-set ex9 --ablations e9_1 e9_2 e9_3 --dataset run_ext_0_1000_61_21 --seeds 20261030 --run-tag physics_v1 --evaluation-grid-shapes 101x101 201x201 401x401
```

只运行采用 `current` 硬边界约束的 e9_3：

```powershell
python scripts/experiments/run.py ex9 e9_3 --dataset run_ext_0_1000_61_21 --seed 20261030 --tag physics_v1 --grids 101 201 401
```
