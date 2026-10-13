# 二维消融实验使用说明

本目录保存二维消融实验的冻结配置、训练、评估和汇总结果。消融实验调用正式的 HMONet 训练与评估代码，但所有运行产物均写入 `experiments/ex1/`，不会与 `outputs/` 中的正式实验混合。

## 1. 配置文件

消融实验采用三层配置：

| 文件 | 作用 |
| --- | --- |
| `baseline/experiment.yaml` | 唯一冻结训练基线，保存模型、损失、采样、学习率和收敛参数 |
| `baseline/evaluate.yaml` | 所有消融实验共用的冻结评估参数，包括测试案例、方法、质量指标和计时设置 |
| `<实验名>/experiment.yaml` | 只定义该消融实验相对基线发生变化的参数及其变体 |
| `configs/ablation_2d.yaml` | 默认数据集、随机种子，以及命令行扫描时使用的默认数值 |

参数优先级为：命令行显式参数 > `<实验名>/experiment.yaml` > `baseline/experiment.yaml`。正常训练配置 `configs/2D/train.yaml` 和正常评估配置 `configs/2D/evaluate.yaml` 的后续修改不会影响消融实验。

冻结基线的主要设置为：

- 数据集与训练网格：`air_t_int_0_1000_41_20`，`41x20`；
- 模型：HMONet，`current` 投影；
- 最大 epoch：30000，batch size：64；
- 最佳模型指标：固定网格 `dataset_mse`；
- 初始学习率：`2e-4`，预热1000次，每1000次乘以0.9，最低学习率 `1e-5`；
- 训练采样：`uniform`，每批共享400个随机点；
- 目标插值：`curved_boundary_cell`；
- Jacobian损失：有限差分，权重0.01，阈值 `1e-4`；
- 法向损失：默认关闭；
- 自动收敛、失败平台、非有限值处理等详细参数以 `baseline/experiment.yaml` 为准。

默认随机种子为 `20260802、20260917、20261031`。论文实验建议在命令行明确指定数据集、随机种子和 `run-tag`，避免误用 `configs/ablation_2d.yaml` 中的通用默认数据集。

## 2. 已定义的消融实验

| 实验名 | 研究问题 | 变体或默认扫描值 | 主要比较指标 |
| --- | --- | --- | --- |
| `jacobian_enabled` | Jacobian损失是否改善网格合法性 | `jac_off`、`jac_on(0.01)` | 非法/自相交比例、最小Jacobian、最小Scaled Jacobian、MSE |
| `jacobian_weight` | Jacobian权重如何影响质量与精度 | `0、1e-4、1e-3、1e-2、1e-1` | 非法/自相交比例、Jacobian、MSE、训练成本 |
| `mapping_mode` | 内部投影方式的质量与成本权衡 | `current`、`coons_residual` | MSE、合法性、Scaled Jacobian、推理时间 |
| `sampling_mode` | 自适应采样是否优于均匀采样 | `uniform`、`adaptive` | MSE、非法比例、Jacobian、训练成本 |
| `target_interpolation` | 曲边目标插值是否改善边界附近网格 | `bilinear`、`curved_boundary_cell` | MSE、自相交比例、边界层正交性 |
| `training_sample_count` | 随机训练点数对精度和成本的影响 | `50、100、200、400、800、1200` | MSE、训练时间、epoch时间、合法性 |
| `initial_learning_rate` | 初始学习率敏感性 | `1e-4、2e-4、4e-4、6e-4、1e-3` | 损失/MSE曲线、收敛epoch、训练时间 |
| `normal_loss` | 法向损失是否值得保留 | `normal_off`、`normal_on(0.01)` | 边界层正交性、MSE、训练时间 |
| `point_weighting` | 位置相关的采样点损失权重是否有效 | `position_weighted`、`uniform_weight` | MSE、合法性、Scaled Jacobian、边界层正交性 |

只有表中指定的参数发生变化，其余参数均继承冻结基线。可直接编辑对应的 `<实验名>/experiment.yaml` 新建变体；开始一轮论文实验后，不建议修改已经使用过的冻结基线。

可在命令行覆盖三个数值扫描：

```text
--initial-learning-rates 0.0001 0.0002 0.0004
--jacobian-weights 0 0.0001 0.001 0.01
--training-sample-counts 50 100 200 400
```

## 3. 运行方法

以下命令均在项目根目录执行，Windows PowerShell 与 Linux Bash 均可使用。

### 3.1 先检查，不执行训练或评估

```powershell
python scripts/experiments/ex1/run_ablations_2d.py `
  --ablations jacobian_enabled jacobian_weight mapping_mode `
  --dataset air_t_int_0_1000_41_20 `
  --seeds 20260802 20260917 20261031 `
  --run-tag paper_v1 `
  --dry-run
```

输出中的 `reuse-training` 表示复用已有模型，`reuse` 表示复用对应网格规模的评估，`evaluate` 表示只需新增评估。

### 3.2 训练并评估独立基线

```powershell
python scripts/experiments/ex1/run_baseline_2d.py `
  --dataset air_t_int_0_1000_41_20 `
  --seeds 20260802 20260917 20261031 `
  --run-tag paper_v1 `
  --evaluation-grid-shapes 51x51 `
  --evaluate
```

建议正式汇总前先准备相同数据集和随机种子的基线。其他消融实验只读 baseline，不会覆盖或删除其中的文件。

### 3.3 运行一个消融实验

```powershell
python scripts/experiments/ex1/run_ablation_2d.py `
  --ablation jacobian_weight `
  --dataset air_t_int_0_1000_41_20 `
  --seeds 20260802 20260917 20261031 `
  --run-tag paper_v1 `
  --evaluation-grid-shapes 51x51 `
  --continue-on-error
```

只运行部分变体时增加，例如：

```text
--variants jac_w_1e-4 jac_w_1e-3
```

### 3.4 一次运行多个消融实验

```powershell
python scripts/experiments/ex1/run_ablations_2d.py `
  --ablations jacobian_enabled jacobian_weight mapping_mode sampling_mode `
  --dataset air_t_int_0_1000_41_20 `
  --seeds 20260802 20260917 20261031 `
  --run-tag paper_v1 `
  --evaluation-grid-shapes 51x51 `
  --continue-on-error
```

使用 `--ablations all` 可选择全部九类实验，但缺少训练结果的变体仍会按配置开始训练。

### 3.5 仅执行训练或评估

```text
--train-only       只训练或复用训练，不执行评估
--evaluate-only    必须已有可复用模型，只执行或复用评估
--dry-run          只显示每个模型将采取的动作
--force            忽略对应阶段缓存并覆盖结果
--continue-on-error 某个变体失败后继续处理其他变体
```

正常补充新网格规模时不要使用 `--force`。例如指定 `--evaluation-grid-shapes 51x51 101x101` 时，程序会独立检查两个评估指纹：已有51×51会直接复用，只计算缺失的101×101。

## 4. 复用与保护规则

- 训练指纹包含数据集内容、模型、损失、采样、学习率和收敛配置；不包含 `run-tag` 和评估参数。
- 评估指纹按单个网格规模计算，并包含 checkpoint 内容、评估配置、评估代码、方法、质量指标和设备。
- 参数和文件完整时跳过训练；评估指纹一致时跳过该网格规模的评估。
- 修改评估网格或质量指标不会触发重新训练。
- 不完整训练无法保证随机状态完全恢复，因此默认从头重训。
- baseline 除显式 `--force` 外不得覆盖；其他消融实验检测到 baseline 配置不一致时会完成各变体工作，但拒绝生成包含 baseline 的科学汇总。

每次运行的 `run_manifest.json` 记录训练复用来源，并在 `evaluations.<网格规模>` 下分别记录评估指纹、身份文件、结果文件和复用状态。

## 5. 输出结构

训练、checkpoint 和身份清单仍保存在对应实验目录；所有评估结果集中到短路径目录。实验名使用清晰缩写，例如 `base`、`lr`、`jac`、`mapping`、`samp`、`tgt` 和 `samples`。

```text
experiments/ex1/
├── <实验名>/<数据集>/<变体>/seed_<seed>/<run_tag>/
│   ├── training/
│   ├── checkpoints/
│   ├── tensorboard/
│   ├── resolved_config.yaml
│   ├── evaluation_identities/<网格规模>.json
│   └── run_manifest.json
├── eval/<数据集>/<实验缩写>/<变体>/<seed>/<run_tag>/<网格规模>/<参数值>/
│   ├── results.json
│   ├── TFI/
│   ├── PDE_Winslow/
│   ├── HMONet/
│   └── comparisons/
└── <实验名>/<数据集>/
    ├── aggregate/<run_tag>/<网格规模>/
    └── figures/<run_tag>/<网格规模>/
```


基线评估省略重复的变体层，因此使用 `eval/<数据集>/base/<seed>/...`。评估临时文件统一写入 `experiments/ex1/.eval_tmp/<短标识>/`，结束后自动删除。代码逐文件检查完整路径不得超过 234 字符，即经典 Windows 260 字符上限的 90%。

`important_metrics.csv` 保存该消融最直接回答研究问题的指标，`secondary_metrics.csv` 保存解释精度、质量与成本权衡的指标，其余可读取数值进入 `other_metrics.csv`。根汇总目录只保存网格索引，不生成跨网格混合表。
## 6. 当前已完成训练的实验

截至当前 `air_t_int_0_1000_41_20 / paper_v1` 记录：

- baseline：三个随机种子均完成；
- 完整完成：`jacobian_enabled`、`jacobian_weight`、`mapping_mode`、`sampling_mode`、`target_interpolation`、`training_sample_count`；
- 尚不完整：`initial_learning_rate`（`lr_6e-4`只有一个种子，`lr_1e-3`无完成记录）；
- 尚未训练：`normal_loss`。

### 当前完整消融实验的101×101评估

下面的命令只选择已经完成三个随机种子训练的六类消融实验。它会复用全部训练，新增101×101评估，不会修改已有51×51结果：

```powershell
python scripts/experiments/ex1/run_ablations_2d.py `
  --ablations jacobian_enabled jacobian_weight mapping_mode sampling_mode target_interpolation training_sample_count `
  --dataset air_t_int_0_1000_41_20 `
  --seeds 20260802 20260917 20261031 `
  --run-tag paper_v1 `
  --evaluate-only `
  --evaluation-grid-shapes 101x101 `
  --continue-on-error
```

如需让各消融实验与 baseline 的101×101结果一起比较，先单独补充 baseline：

```powershell
python scripts/experiments/ex1/run_ablation_2d.py `
  --ablation baseline `
  --dataset air_t_int_0_1000_41_20 `
  --seeds 20260802 20260917 20261031 `
  --run-tag paper_v1 `
  --evaluate-only `
  --evaluation-grid-shapes 101x101 `
  --continue-on-error
```

迁移旧评估目录时可运行：

```powershell
python tools/migrations/migrate_short_evaluation_paths.py --dry-run
python tools/migrations/migrate_short_evaluation_paths.py
```

迁移会移动原始评估目录并更新清单引用，但不会修改 checkpoint、训练指纹、评估指纹或 `results.json` 内容。

