# 二维训练命令行使用说明

对应配置：[train.yaml](train.yaml)；评估说明：[evaluate_cli.md](evaluate_cli.md)。
入口：`scripts/train/train_2d.py`。以下选项依据当前入口、配置合并和训练逻辑核对，按常用程度排列。

运行前先执行：

```powershell
conda activate MGNet
cd E:\projects\HMONet
```

参数清单中的选项追加在 `python scripts/train/train_2d.py` 后使用；每一项都是独立选项，无需全部同时输入。完整命令示例可直接复制。

- 命令行覆盖仅对本次运行生效，不会回写 YAML；省略选项时使用 YAML 或代码默认值。
- 布尔开关直接写名称，后面不加 `true/false`；`--no-` 前缀关闭。路径包含空格时加双引号。
- 公共参数位于 `training`，模型专属参数位于 `hmonet_training`、`meshonet_training` 等段；平台开关优先级为命令行 > 模型专属段 > 公共段。
- 以下示例按当前 air_t 数据集的 `21×20` 点数编写。训练前应已有数据集；训练入口不会生成数据集。

## 目录

- [1. 完整命令示例](#examples)
- [2. 数据集、模型、网格和基本训练参数](#basics)
- [3. 随机采样、模型损失和内部映射](#sampling)
- [4. 固定网格 MSE 和最佳模型选择](#mse)
- [5. 收敛判定与平台开关](#convergence)
- [6. 学习率、梯度和非有限值处理](#optimizer)
- [7. 输出位置、TensorBoard 和文件](#outputs)
- [8. 帮助、兼容参数和只在 YAML 中修改的设置](#other-options)

<a id="examples"></a>

## 1. 完整命令示例

按当前 YAML 训练。当前配置为 HMONet、内插和外推分别训练，最大 100000 个 epoch，启用的收敛条件可提前停止。

```powershell
python scripts/train/train_2d.py --config configs/2D/train.yaml
```

训练两种模型、两类数据集，共四次独立训练并分别输出。每次训练都会重新初始化模型。

```powershell
python scripts/train/train_2d.py --model both --dataset-mode both
```

只训练内插 HMONet，使用 current 内部映射。

```powershell
python scripts/train/train_2d.py --model hmonet --dataset-mode interpolation --hmonet-mapping-mode current --epochs 100000 --batch-size 64 --seed 20260802 --device cuda
```

只训练外推 MeshONet，修改内部随机点数及内点/边界损失权重。

```powershell
python scripts/train/train_2d.py --model meshonet --dataset-mode extrapolation --interior-sample-count 400 --alpha 1.0 --beta 5.0
```

指定一个数据集目录，训练 HMONet 的 Coons 残差模式。此时 `--dataset` 直接决定读取的数据，`--dataset-mode` 用于本次训练的模式标记。

```powershell
python scripts/train/train_2d.py --dataset data/raw/2D/air_t_int_0_1000_21_20 --dataset-mode interpolation --model hmonet --hmonet-mapping-mode coons_residual --residual-regularization-weight 0.0
```

两种模型都采用每个 batch 共享的 400 个均匀随机内部点，并且每个 epoch 测量全数据集固定内部节点的 MSE。

```powershell
python scripts/train/train_2d.py --model both --dataset-mode interpolation --fixed-grid-count 0 --global-random-count 400 --interior-sample-count 400 --calculate-dataset-mse --dataset-mse-interval 1 --best-model-metric dataset_mse
```

仅用固定网格 MSE 收敛条件：连续 30 次测量不超过 `1e-6` 时停止，同时关闭其他提前停止条件。

```powershell
python scripts/train/train_2d.py --model hmonet --dataset-mode interpolation --calculate-dataset-mse --check-mse-convergence --mse-tolerance 1e-6 --mse-consecutive-epochs 30 --no-check-training-loss-convergence --no-check-max-point-error --no-check-p99-point-error --no-check-plateau-stability --no-check-plateau-best-improvement
```

短流程检查：两种模型各训练一个 epoch，关闭全部提前停止开关，保留固定网格 MSE，输出到独立目录。该命令会实际训练并保存结果。

```powershell
python scripts/train/train_2d.py --model both --dataset-mode interpolation --epochs 1 --batch-size 64 --no-check-training-loss-convergence --no-check-mse-convergence --no-check-max-point-error --no-check-p99-point-error --no-check-plateau-stability --no-check-plateau-best-improvement --no-tensorboard-enabled --output-root outputs/cli_smoke/training/2D
```

<a id="basics"></a>

## 2. 数据集、模型、网格和基本训练参数

- `--model both`：覆盖 `model.name`；可选 `hmonet`、`meshonet`、`both`。`both` 依次训练 HMONet、MeshONet。
- `--dataset-mode both`：覆盖 `training.dataset`；可选 `interpolation`、`extrapolation`、`both`。未指定 `--dataset` 时从 YAML 的 `datasets` 读取对应路径；`both` 按内插、外推分别训练，不合并为一个训练集。
- `--dataset data/raw/2D/air_t_int_0_1000_21_20`：直接指定一个数据集目录，替代 YAML 路径选择；也可指定该目录内的 `dataset.npy`。相对路径相对于运行命令的当前目录，推荐在项目根目录运行。
- `--nx 21`：可选的一致性检查；训练规模自动读取数据集，若显式值与数据集不一致则中止。
- `--ny 20`：可选的一致性检查；训练规模自动读取数据集，若显式值与数据集不一致则中止。
- `--epochs 100000`：覆盖 `training.num_epochs`，修改最大 epoch 数，必须大于 0；一个 epoch 完整遍历当前数据集一次。
- `--num-epochs 100000`：同样覆盖 `training.num_epochs`；建议与 `--epochs` 二选一，两者同时出现时 `--epochs` 优先，与命令书写顺序无关。
- `--batch-size 64`：覆盖 `training.batch_size`，修改每个 batch 的几何样本数，必须大于 0；不是单个几何的随机点数。
- `--lr 0.0004`：覆盖本次所选每个模型各自训练段中的 `learning_rate`；不传时各模型独立读取自己的 YAML 值。
- `--seed 20260802`：覆盖 `training.seed`，控制模型初始化、数据集打乱和随机点采样；当前 YAML 与旧项目 air_t 使用相同种子。
- `--device cuda`：覆盖 `training.device`；支持 `auto`、`cpu`、`cuda`、`cuda:0` 等。`auto` 优先使用可用 CUDA，否则使用 CPU。
- `--config configs/2D/train.yaml`：指定训练 YAML，省略时使用此路径；相对路径相对于当前工作目录。
- `--output-root outputs/training/2D`：指定训练日志输出根目录；这是入口参数，不是 `training` 下的配置项。对应 checkpoint 和 TensorBoard 的路径推导见第 7 节。

当前 YAML 的数据集目录分别是 `data/raw/2D/air_t_int_0_1000_21_20` 和 `data/raw/2D/air_t_ext_0_1000_21_20`。MeshONet 读取 `dataset.npy` 中的边界采样点和 PDE 网格；HMONet 还读取对应的 `boundary_geometry.npy` 并恢复参数边界类。两个文件的样本索引必须对应。

训练网格直接读取 `dataset.npy` 中的 `grid_shape`；`--nx/--ny` 不会重采样，只用于检查。更换训练网格时，只需选择或生成相应网格规模的数据集。使用显式 `--dataset` 时只会读取这一个数据集；此时即使 `--dataset-mode both`，也不会额外读取另一类数据。

<a id="sampling"></a>

## 3. 随机采样、模型损失和内部映射

每个 epoch 打乱样本顺序。每个 batch 生成一组新的计算域均匀随机点，batch 内所有几何共享这些查询点并批量前向计算；PDE 网格插值提供随机查询点的目标坐标。epoch 损失按各个有效 batch 的实际样本数加权平均。

- `--sampling-mode uniform|adaptive`：选择全局均匀随机采样或曲率-边界距离自适应采样。
- `--global-random-count 400`：uniform 模式的点数；同一 batch 共享一组随机点。
- `--curvature-adaptive-count 400`：adaptive 模式中每个几何样本的点数。
- `--adaptive-candidate-multiplier 10`：自适应候选点数相对最终点数的倍数。
- `--adaptive-curvature-quantile 0.95`、`--adaptive-curvature-power 1.0`：曲率归一化分位数和曲率权重指数。
- `--adaptive-distance-sigma-min 0.02`、`--adaptive-distance-sigma-max 0.10`：高、低曲率边界点的距离核宽度。
- `--adaptive-uniform-probability 0.05`：均匀概率底值，保留远场覆盖。
- `--fixed-grid-count 0`：覆盖 `hmonet_sampling.fixed_grid_count`，修改 HMONet 额外使用的固定网格内部节点数量；0 关闭此分支。开启时每个 batch 从固定内部节点中随机选一组索引，各样本共享；上限为 `(nx-2)*(ny-2)`。
- `--interior-sample-count 400`：覆盖 `meshonet_training.interior_sample_count`，修改 MeshONet 每个样本的内部随机点数；必须大于 0，同一 batch 共享一组随机点。
- `--alpha 1.0`：覆盖 `meshonet_training.alpha`，修改 MeshONet 内部坐标损失权重。
- `--beta 5.0`：覆盖 `meshonet_training.beta`，修改 MeshONet 边界坐标损失权重。
- `--hmonet-mapping-mode current`：覆盖 `model.hmonet.hmonet_mapping_mode`；训练入口只支持 `current`、`coons_residual`。分别使用当前投影映射或 Coons 基础映射加网络残差。
- `--residual-regularization-weight 0.0`：覆盖 `model.hmonet.residual_regularization_weight`，修改 Coons 残差位移的正则权重；0 关闭，值必须非负，仅 Coons 残差模式具有对应位移项。
- `--target-interpolation curved_boundary_cell`：使用精确参数边界位移的 IDW 修正训练目标；也可设为 `bilinear`。
- `--idw-neighbors -1`：IDW 邻居数；`-1` 使用全部支持点（默认），`32` 启用 K=32 局部近邻模式。
- `--idw-query-chunk-size 128`：IDW 查询点分块；二维默认 128，三维配置默认 512。
- `--idw-support-chunk-size 2048`：IDW 支持点分块；较大的分块可减少 Python 循环和 GPU 内核启动次数。
- `--jacobian-loss-enabled`：设置 `hmonet_training.jacobian_loss_enabled=true`，启用 HMONet 内部点的正 Jacobian 约束。
- `--no-jacobian-loss-enabled`：设置 `hmonet_training.jacobian_loss_enabled=false`，关闭上述约束，当前 YAML 默认关闭。
- `--trunk-projection-coefficients-enabled` / `--no-trunk-projection-coefficients-enabled`：开启或关闭主干几何系数 MLP；开启后主干与分支的四个系数逐位相乘。
- `--projection-softplus-enabled` / `--no-projection-softplus-enabled`：几何系数使用 Softplus 或原 Sigmoid 激活。
- `--boundary-self-attention-enabled` / `--no-boundary-self-attention-enabled`：开启或关闭四条边界的联合自注意力编码。
- `--projection-mlp-hidden-dim 64`：主干几何系数 MLP 的隐藏层宽度。
- `--attention-dim 64 --attention-heads 4 --attention-layers 2 --attention-feedforward-dim 128 --attention-dropout 0`：覆盖 HMONet 自注意力结构。
- `--jacobian-loss-weight 0.01`：覆盖 `hmonet_training.jacobian_loss_weight`，修改 Jacobian 约束权重。
- `--jacobian-minimum 1e-4`：覆盖 `hmonet_training.jacobian_minimum`，修改有向 Jacobian 的最低目标值，低于该值时产生惩罚。
- `--boundary-normal-loss-enabled`: enable inward grid-direction alignment with exact inward unit normals: `mean((1 - dot)^2)`. In 2D the target comes from curve tangents; in 3D from surface tangent cross products. PDE mesh displacements determine only the inward sign. Without parametric geometry, sampled boundary tangents are used instead.
- `--no-boundary-normal-loss-enabled`：设置 `hmonet_training.boundary_normal_loss_enabled=false`，关闭上述约束，当前 YAML 默认关闭。
- `--boundary-normal-loss-weight 0.01`：覆盖 `hmonet_training.boundary_normal_loss_weight`，修改法向约束权重。
- `--boundary-normal-sample-count 64`：覆盖 `hmonet_training.boundary_normal_sample_count`，修改每个样本用于法向约束的边界配对采样点总数。
- `--boundary-normal-offset 0.01`：覆盖 `hmonet_training.boundary_normal_offset`，修改从边界向计算域内部偏移的参数距离，取值必须位于 `(0,0.5)`。

HMONet 的固定点数量和当前模式对应的随机点数量必须非负且至少一个大于 0。固定节点训练分支与“全数据集固定网格 MSE”独立：`--fixed-grid-count 0` 不会关闭 MSE 监控。

训练 HMONet 时会传入参数边界类，直接计算精确投影点；不需要另外设置曲线输入开关。网络内部映射和评估阶段的边界后处理是不同设置；训练入口没有 `--hmonet-boundary-projection` 或 `--meshonet-boundary-projection`。

<a id="mse"></a>

## 4. 固定网格 MSE 和最佳模型选择

这里的 `dataset_mse` 在全部几何样本的固定内部网格节点上测量，排除边界节点，使用 `2 * mean((prediction-target)**2)`。二维时等价于对每个点的 `dx²+dy²` 求平均；不包含 HMONet 训练损失的空间权重或其他正则项。

- `--best-model-metric dataset_mse`：覆盖 `training.best_model_metric`；可选 `dataset_mse`、`training_loss`。前者按固定网格 MSE 选最佳模型，后者按 epoch 训练损失选择。
- `--calculate-dataset-mse`：设置 `training.calculate_dataset_mse=true`，按配置频率测量固定网格 MSE。
- `--no-calculate-dataset-mse`：设置 `training.calculate_dataset_mse=false`；是否仍在 epoch 内测量还受最佳模型选择和点误差检查影响，见下文。
- `--dataset-mse-interval 1`：覆盖 `training.dataset_mse_interval`，修改 MSE 测量间隔，必须大于 0；1 表示每个 epoch 测量所有样本，较大值仍包含第 1 轮和达到最大轮数时的最后一轮。
- `--check-mse-convergence`：设置 `training.check_mse_convergence=true`，启用固定网格 MSE 连续达标的收敛条件。
- `--no-check-mse-convergence`：设置 `training.check_mse_convergence=false`，关闭该收敛条件，不等于关闭 MSE 计算。
- `--mse-tolerance 1e-6`：覆盖 `training.mse_tolerance`，设置固定网格 MSE 达标阈值，必须非负。
- `--mse-consecutive-epochs 30`：覆盖 `training.mse_consecutive_epochs`，设置连续达标的 MSE **测量次数**，必须大于 0；间隔不为 1 时不是连续的训练 epoch 数。
- `--check-max-point-error`：设置 `training.check_max_point_error=true`，启用固定内部节点最大欧氏距离误差的收敛检查。
- `--no-check-max-point-error`：设置 `training.check_max_point_error=false`，关闭最大点误差检查。
- `--max-point-error-tolerance 1e-3`：覆盖 `training.max_point_error_tolerance`，修改最大点误差阈值；这是坐标距离，不是平方误差。
- `--check-p99-point-error`：设置 `training.check_p99_point_error=true`，启用固定内部节点欧氏距离误差的第 99 百分位检查。
- `--no-check-p99-point-error`：设置 `training.check_p99_point_error=false`，关闭 P99 点误差检查。
- `--p99-point-error-tolerance 1e-3`：覆盖 `training.p99_point_error_tolerance`，修改 P99 点误差阈值；这是坐标距离，不是平方误差。

单独 `--no-calculate-dataset-mse` 不保证关闭 epoch 内的 MSE：当 `best_model_metric=dataset_mse` 时仍按间隔测量；启用任意点误差检查时每个 epoch 都会测量。若需关闭训练过程中的测量，应同时选择 `--best-model-metric training_loss`、关闭 MSE 收敛和两种点误差检查，并关闭基于 MSE 的平台判定或改用 `--plateau-metric train_loss`。训练收尾仍会测量最终模型及最佳模型的固定网格 MSE 并记录。

<a id="convergence"></a>

## 5. 收敛判定与平台开关

- `--check-training-loss-convergence`：设置 `training.check_training_loss_convergence=true`，启用训练损失连续达标检查。
- `--no-check-training-loss-convergence`：设置 `training.check_training_loss_convergence=false`，关闭训练损失连续达标检查。
- `--training-loss-tolerance 1e-6`：覆盖 `training.training_loss_tolerance`，修改 epoch 训练损失阈值，必须非负。
- `--training-loss-consecutive-epochs 50`：覆盖 `training.training_loss_consecutive_epochs`，设置训练损失连续达标所需的 epoch 数，必须大于 0。
- `--check-plateau-stability`：开启当前训练模型的平台波动稳定性检查；覆盖公共及该模型的 `check_plateau_stability`。
- `--no-check-plateau-stability`：关闭当前训练模型的平台波动稳定性检查。
- `--check-plateau-best-improvement`：开启当前训练模型的平台最优值无显著改善检查；覆盖公共及该模型的 `check_plateau_best_improvement`。
- `--no-check-plateau-best-improvement`：关闭当前训练模型的平台最优值无显著改善检查。
- `--plateau-metric train_loss`：覆盖 `training.plateau_metric`；可选 `train_loss`、`mse`，分别使用训练损失或固定内部网格 MSE 检查平台。
- `--convergence-start-epoch 3000`：覆盖 `training.convergence_start_epoch`，设置所有自动收敛和失败判定开始检查并累计耐心的最早 epoch。
- `--plateau-window 200`：覆盖 `training.plateau_window`，设置成功平台和失败平台共用的滑动窗口，至少为 2。
- `--fluctuation-quantiles 0.1 0.9`：覆盖 `training.fluctuation_quantiles`，设置波动统计的上下分位数；使用两个空格分隔的数字，满足 `0 <= lower < upper <= 1`。
- `--fluctuation-relative-tolerance 0.5`：覆盖 `training.fluctuation_relative_tolerance`，设置相对波动 `(Q_upper-Q_lower)/max(abs(median),1e-12)` 的允许上限。
- `--best-improvement-relative-tolerance 0.005`：覆盖 `training.best_improvement_relative_tolerance`，设置有效最优值的最小显著改善比例；0.005 表示 0.5%。
- `--success-plateau-patience 1000`：覆盖 `training.success_plateau_patience`；记录连续无显著改善的检查次数，该计数只在出现显著改善时归零。达到该耐心并接近历史最佳后，`training.plateau_stable_consecutive_checks` 控制的短窗口连续稳定分支或长期波动稳定分支任一通过即可停止。
- `--plateau-log-interval 100`：覆盖 `training.plateau_log_interval`，设置平台详细状态按 epoch 输出的间隔；不改变检查频率或每个 epoch 的训练进度输出。
- --failure-plateau-enabled / --no-failure-plateau-enabled：启用或关闭失败平台停止。
- --failure-plateau-patience 2000：失败条件完整通过时加一，任一条件失败时归零。
- --failure-plateau-metric-ratio 1.5：当前窗口中位数至少达到历史最佳窗口中位数的该倍数时，才视为明显较差。
- --failure-plateau-relative-fluctuation 0.15：失败平台允许的最大相对波动。
- --failure-plateau-min-lr-updates 2：失败停止前至少完成的学习率更新次数。
- `--check-convergence`：兼容名称，强制开启 `training.check_training_loss_convergence`；并非所有收敛条件的总开关。
- `--no-check-convergence`：兼容名称，强制关闭 `training.check_training_loss_convergence`；不会关闭 MSE、点误差或平台检查。

`--check-convergence` 与 `--no-check-convergence` 互斥。如果同时给出兼容名称和 `--check-training-loss-convergence` 的正/反向开关，兼容名称最终优先；建议使用明确的训练损失开关并避免混用。

MSE、最大点误差和 P99 点误差中，所有已启用的条件共同达标后触发指标收敛；训练损失连续达标、平台收敛则可以分别触发提前停止。两个平台检查均关闭时才完全关闭平台判断。平台使用 MSE 且测量间隔大于 1 时，窗口长度和耐心按有数据的检查次数累计。

关闭所有提前停止条件需要关闭训练损失、MSE、最大点误差、P99 点误差及两个平台开关；第 1 节的短流程命令给出了完整写法。

<a id="optimizer"></a>

## 6. 学习率、梯度和非有限值处理

- `--lr-warmup-epochs 1000`：覆盖本次所选每个模型各自的 `lr_warmup_epochs`；这里是保持初始值，不是从零逐步增大学习率。
- `--lr-update-interval 500`：覆盖本次所选每个模型各自的 `lr_update_interval`，设置保持阶段结束后的更新间隔。
- `--lr-decay-gamma 0.8`：覆盖本次所选每个模型各自的 `lr_decay_gamma`，满足 `0 < gamma <= 1`。
- `--lr-stop-below 1e-5`：覆盖本次所选每个模型各自的 `lr_stop_below`；最后一次更新后允许低于该阈值。
- `--gradient-clip-norm 1.0`：覆盖 `training.gradient_clip_norm`，设置梯度范数裁剪上限，必须大于 0；不接受 `null` 字符串，若要关闭裁剪，在 YAML 中设为 `null`。
- `--skip-nonfinite-loss`：设置 `training.skip_nonfinite_loss=true`；loss 或梯度非有限时打印警告、跳过当前 batch 的优化和损失累计。
- `--no-skip-nonfinite-loss`：设置 `training.skip_nonfinite_loss=false`；发现上述非有限值时抛出异常，停止训练。

例如保持 1000 轮、间隔 500 轮，首次衰减发生在第 1500 轮完成后，新学习率用于下一轮。关闭梯度裁剪后仍会检查梯度是否有限；若一个 epoch 的所有 batch 都无效，则会报错。

<a id="outputs"></a>

## 7. 输出位置、TensorBoard 和文件

- `--tensorboard-enabled`：设置 `training.tensorboard_enabled=true`，启用训练目录内日志及独立目录中的 TensorBoard 镜像。
- `--no-tensorboard-enabled`：设置 `training.tensorboard_enabled=false`，关闭这两处 TensorBoard 写入，不影响其他训练文件。

默认输出位置如下。数据集名称本身包含 `_int` 或 `_ext`，所以不再增加 `interpolation` 或 `extrapolation` 目录；`<映射>` 为 `current` 或 `coons_residual`。HMONet 的映射目录位于日期之前，MeshONet 没有映射目录；日期格式为 `YYYYMMDD_HHMM`，同一分钟重复训练使用 `_02`、`_03` 等后缀。

```text
outputs/
├── training/2D/<数据集>/
│   ├── hmonet/<映射>/<日期>/
│   └── meshonet/<日期>/
├── checkpoints/2D/<数据集>/
│   ├── HMONet/<映射>/<日期>/
│   └── MeshONet/<日期>/
└── tensorboard/2D/<数据集>/
    ├── hmonet/<映射>/<日期>/
    └── meshonet/<日期>/
```

每个 `training` 日期目录包含：

- `config.json`：本次训练合并命令行后的有效配置。
- `training_history.csv`、`training_loss.png`：逐 epoch 训练记录和损失曲线。
- `mse_history.csv`、`dataset_mse.png`：固定内部节点 MSE 的测量轮次、数值和曲线。
- `training_summary.json`、`training_summary.csv`：完成轮数、耗时、停止原因、最佳轮次、最佳指标值等摘要。
- `tensorboard/events.out.tfevents.*`：启用 TensorBoard 时生成；相同内容同时写入独立的 `outputs/tensorboard` 日期目录。

每个 `checkpoints` 日期目录包含：

- `best.pt`、`final.pt`：模型权重、配置、网格大小、模型名称和所选轮次。
- `best_model.pth`、`final_model.pth`：纯模型 `state_dict` 参数文件。
- `config.json`、`config.yaml`：本次有效配置。
- `training_summary.json`、`training_summary.csv`：本次训练摘要副本。
- `checkpoint_metadata.json`、`model_parameters.json`：模型类型、内部映射、权重位置及结构配置等元数据。

每个 epoch 显示 batch 进度、损失和已测量的 MSE；每次训练结束输出该模型的轮数、耗时、最佳指标和保存位置，全部选择项完成后再输出汇总。当前入口没有断点续训参数，也不保存优化器状态；每次命令启动一次新训练。

自定义输出时，程序按 `Path(--output-root).parent.parent` 作为总输出根，再追加 `checkpoints` 和 `tensorboard`。建议保持 `<自定义根>/training/2D` 的形式，例如第 1 节的 `outputs/cli_smoke/training/2D` 对应 `outputs/cli_smoke/checkpoints/2D` 和 `outputs/cli_smoke/tensorboard/2D`。

查看独立 TensorBoard 日志：

```powershell
tensorboard --logdir outputs/tensorboard/2D
```

<a id="other-options"></a>

## 8. 帮助、兼容参数和只在 YAML 中修改的设置

- `-h`：显示训练入口所有命令行参数并退出，不运行训练。
- `--help`：与 `-h` 相同；完整命令如下。
- `--curvature-adaptive-count N`：设置 adaptive 模式每个样本的内部点数；需同时使用 `--sampling-mode adaptive`。

```powershell
python scripts/train/train_2d.py --help
```

二维使用 `scripts/train/train_2d.py`，三维使用 `scripts/train/train_3d.py` 和 `configs/3D/train.yaml`。

网络层数、隐藏层宽度、特征维度等结构参数在 YAML 的 `model.common` 和各模型段中修改，没有逐项命令行覆盖。两个数据集的默认路径在 YAML 的 `datasets` 修改。不要把评估入口的 `--grid-shapes`、`--methods`、`--checkpoint`、`--hmonet-geometry-mode`、`--curve-kernel-backend` 等选项追加到训练命令中；当前训练解析器不支持这些选项。

训练映射模式不接受评估入口中的 `both` 或 `coons` 别名。若要训练两种 HMONet 映射，应分别执行 `--hmonet-mapping-mode current` 和 `--hmonet-mapping-mode coons_residual`。


## Train followed by evaluation

Run from the project root in the MGNet environment:

```powershell
python scripts/train/train_evaluate.py --dimension 2
```

The pipeline follows train.yaml, trains all selected models/datasets, then evaluates
only those dataset cases with their newly saved best.pt files. TFI and PDE are
included alongside the models trained in this run. Failures stop the pipeline.
The original training, checkpoint and evaluation output paths are preserved.
Dataset paths and grid sizes must match existing files, and evaluate.yaml must
contain a case matching each dataset directory name.

```powershell
# Train both networks; train.yaml chooses interpolation/extrapolation/both.
python scripts/train/train_evaluate.py --dimension 2 --train-args "--model both"
# Check dataset/grid/case compatibility without starting training.
python scripts/train/train_evaluate.py --dimension 2 --check-only
# Explicitly use the existing ESA interpolation dataset with a 20x20 grid.
python scripts/train/train_evaluate.py --dimension 2 --train-args "--dataset data/raw/2D/esa_int_0_1000_20_20 --dataset-mode interpolation --model both"
# Optional final checkpoint and evaluation overrides.
python scripts/train/train_evaluate.py --dimension 2 --checkpoint final --evaluate-args "--no-compute-grid-quality"
```

Use --train-config / --evaluate-config for alternate YAML files. Use forward slashes
in paths inside quoted --train-args and --evaluate-args. Evaluation case, method,
mapping and checkpoint selection are automatic and cannot be overridden there.
The same entry supports --dimension 3 with the 3D configuration files.


## PI-HMONet 与四模型选择

现支持 `attentionpihmonet`；`both` 保持 HMONet＋MeshONet，`pi` 依次训练三个 PI 模型，`all` 选择全部五个模型。
PI-HMONet 训练使用 `--pihmonet-mapping-mode current/coons_residual`；评估还可设为 `both`。
PI-HMONet 专属覆盖包括 pihmonet-pde-loss-weight、
pihmonet-interior-sample-count、pihmonet-fd-step、
pihmonet-jacobian-loss-weight 和 pihmonet-jacobian-minimum。
PI-MeshONet 同样支持 `--pimeshonet-*` 前缀的前四项；专属覆盖优先于原有 `--pi-*` 公共覆盖。
评估支持 `--pihmonet-checkpoint`、`--pihmonet-boundary-projection` 与 `--methods tfi,pde,pihmonet`。
训练评估串联时，模型和映射由 `--train-args` 指定，评估自动使用本次模型。
完整流程、公式及二维三维命令示例见项目中的 `scripts/train/pi_models.md`。


## Jacobian 损失求导方式（二维、三维）

HMONet 的 `jacobian_loss_enabled` 开关和权重含义不变；默认采用有限差分。
`--jacobian-method finite_difference` 使用有限差分，`--jacobian-method autograd` 恢复原自动微分。
`--jacobian-fd-step 0.001` 设置计算域步长，仅在有限差分模式生效。
内部点使用中心差分，边界附近使用二阶单侧差分，查询点不越出计算域。
二维将中心与四个邻点、三维将中心与六个邻点合并前向；主坐标损失只使用中心预测。
自动微分模式仍对完整硬约束输出求导。两种方式均允许损失反向传播到网络参数。
PI-HMONet 当前仍仅优化 PDE 损失，不因这个默认值改变而自动增加 Jacobian 损失。

```powershell
python scripts/train/train_2d.py --model hmonet --jacobian-loss-enabled --jacobian-method finite_difference --jacobian-fd-step 0.001
python scripts/train/train_3d.py --model hmonet --jacobian-loss-enabled --jacobian-method autograd
python scripts/train/train_evaluate.py --dimension 2 --train-args "--model all --dataset-mode both --jacobian-method finite_difference --jacobian-fd-step 0.001"
```

损失仍为 `mean(relu(jacobian_minimum - det(J))**2)`；差分步长会影响近似精度，不能保证与自动微分在尖点或分段接头处完全一致。

### PI-HMONet Jacobian 损失

PI-HMONet 默认启用有限差分 Jacobian 合法性损失，并复用 PDE 模板点，不增加额外网络前向：

L = L_PDE + 0.01 * L_Jacobian

可使用 --pihmonet-jacobian-loss-weight 0.01 修改权重，使用
--pihmonet-jacobian-minimum 1e-4 修改最小目标值，或使用
--no-pihmonet-jacobian-loss-enabled 临时关闭。

## PI 两模型联合训练与评估

模型选择参数 --model pi 表示依次训练 PI-MeshONet、PI-HMONet 和 Attention-PI-HMONet。
在 train_evaluate.py 中使用时，训练完成后自动使用本次训练产生的两个
checkpoint，按 TFI、PDE、PI-MeshONet、PI-HMONet 的顺序评估：

    python scripts/train/train_evaluate.py --dimension 2 --train-args "--model pi --dataset-mode both" --evaluate-args "--model pi"

--model both 仍表示 HMONet 与 MeshONet；--model all 表示全部五个网络。

### Attention-PI-HMONet

正式统一入口使用模型名 `attentionpihmonet`。模型与四个已有模型共用收敛判断、日志、TensorBoard、检查点和训练摘要：

```powershell
python scripts/train/train_2d.py --model attentionpihmonet --dataset-mode interpolation
```

内部 PDE 可用 `--attentionpihmonet-pde-backend autograd`（默认）或
`finite_difference`；边界层 PDE 与 Jacobian 始终使用自动微分。
锚点损失默认关闭，可用 `--attentionpihmonet-anchor-loss-weight` 开启。
主要损失覆盖参数为：

- `--attentionpihmonet-normalized-pde-loss-weight`
- `--attentionpihmonet-jacobian-loss-weight`
- `--attentionpihmonet-smoothness-loss-weight`
- `--attentionpihmonet-boundary-pde-loss-weight`
- `--attentionpihmonet-boundary-jacobian-loss-weight`
- `--attentionpihmonet-boundary-physics-sample-count`
- `--attentionpihmonet-boundary-physics-offset`
- `--attentionpihmonet-boundary-physics-tangential-margin`

训练后直接评估：

```powershell
python scripts/train/train_evaluate.py --dimension 2 --train-args "--model attentionpihmonet --dataset-mode interpolation" --evaluate-methods tfi,pde,attentionpihmonet
```




