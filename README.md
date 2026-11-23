# HMONet

HMONet 是面向二维和三维结构化网格生成的研究项目。项目使用参数曲线和参数曲面描述精确边界，提供数据集生成、模型训练、网格评估和消融实验流程。

当前实现包括：

- HMONet、MeshONet、PI-MeshONet、QG-HMONet 等模型；
- 二维参数曲线边界和三维参数曲面边界；
- TFI 与 Winslow PDE 网格基线；
- 网格质量、非法单元、边界误差和生成时间评估；
- 二维消融实验与多随机种子汇总。

## 项目结构

```text
HMONet/
├── configs/        # 二维和三维训练、评估配置
├── data/           # 本地数据集，不纳入 Git
├── experiments/    # 消融实验配置
├── scripts/        # 训练、评估、实验和可视化入口
├── src/            # 数据、模型、损失、数值方法、评估和训练引擎
├── tests/          # 自动化测试
├── docs/           # 补充说明
├── reports/        # 自动生成或冻结的科研结果报告
├── requirements.txt      # 运行依赖
└── requirements-dev.txt  # 完整测试依赖
```

## 环境配置

推荐使用 Python 3.10，并根据计算设备安装相应版本的 PyTorch。

```bash
conda create -n HMONet python=3.10
conda activate HMONet
pip install -r requirements.txt
```

在 Windows 上也可以使用现有的 `MGNet` 环境：

```powershell
conda activate MGNet
cd E:\projects\HMONet
```

## 从零复现：九类外推数据集与正式实验

下面的流程假设当前没有数据集、训练 checkpoint 或评估结果；所有命令都从项目根目录运行。数据集写入 `data/raw/2D/` 或 `data/raw/3D/`，训练与评估结果写入 `outputs/`。推荐通过统一入口 `scripts/train/train_evaluate.py` 完成一个数据集的“训练 → 评估”闭环；它会自动生成 TFI 和 Winslow PDE 基线，并评估本次训练出的 HMONet、MeshONet、PI-MeshONet。

`--models hmonet meshonet pimeshonet` 是论文主对比组；需要同时测试 QG-HMONet 时，在其后追加 `qghmonet`，或使用 `--models all`。二维纯数字网格会展开为方格；三维推荐明确写成 `n_x × n_y × n_z`，避免纯数字被解释成立方网格。

### 二维：九类外推数据集

| 类别 | 生成族 | 输出数据集 | 默认外推评估参数 |
| --- | --- | --- | --- |
| 2D-1 | `circle_airfoil_t` | `air_t_ext_1_500_41_41` | `0.62 0.65` |
| 2D-2 | `eccentric_semiannulus` | `esa_ext_1_500_41_41` | `0.72 0.75` |
| 2D-3 | `rect_top_semiellipse` | `rse_ext_1_500_41_41` | `0.42 0.46 0.50 0.55 0.60` |
| 2D-4 | `rounded_u_notch_compact` | `run_ext_1_500_41_41` | `1.52 1.55` |
| 2D-5 | `smooth_constricted_channel` | `scc_ext_1_500_41_41` | `0.42 0.45 0.47` |
| 2D-6 | `variable_width_s_channel` | `vsc_ext_1_500_41_41` | `0.85 0.90 0.95` |
| 2D-7 | `symmetric_convex_tri` | `tri_p_ext_1_500_41_41` | `0.42 0.48` |
| 2D-8 | `gear_spline` | `gsp_a_ext_1_500_41_41` | `0.19 0.20` |
| 2D-9 | `four_convex_arcs` | `fca_a_ext_1_500_41_41` | `125 130` |

先逐类生成训练数据：

二维九类边界的外推与内插完整命令（包括参数范围、样本数、PDE 网格和求解器参数）见 [data/2D/generation_commands.md](data/2D/generation_commands.md)。

然后逐类训练并评估三种主对比模型：

```powershell
python scripts/train/train_evaluate.py --dimension 2 --dataset air_t_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.62 0.65 --seed 20261030 --run-tag v1
python scripts/train/train_evaluate.py --dimension 2 --dataset esa_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.72 0.75 --seed 20261030 --run-tag v1
python scripts/train/train_evaluate.py --dimension 2 --dataset rse_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.42 0.46 0.50 0.55 0.60 --seed 20261030 --run-tag v1
python scripts/train/train_evaluate.py --dimension 2 --dataset run_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 1.52 1.55 --seed 20261030 --run-tag v1
python scripts/train/train_evaluate.py --dimension 2 --dataset scc_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.42 0.45 0.47 --seed 20261030 --run-tag v1
python scripts/train/train_evaluate.py --dimension 2 --dataset vsc_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.85 0.90 0.95 --seed 20261030 --run-tag v1
python scripts/train/train_evaluate.py --dimension 2 --dataset tri_p_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.42 0.48 --seed 20261030 --run-tag v1
python scripts/train/train_evaluate.py --dimension 2 --dataset gsp_a_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.19 0.20 --seed 20261030 --run-tag v1
python scripts/train/train_evaluate.py --dimension 2 --dataset fca_a_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 125 130 --seed 20261030 --run-tag v1
```

### 三维：九类外推数据集

九类三维边界的两个端面分别采用201×201节点二维Winslow求解，再以精确Coons边界加三次样条位移定义连续参数面。四侧面在原参数域使用TFI，保留原解析几何；不做参数重分布、不复用上下端面求值。体内仍求解三维PDE。run_e默认41×41×11，其余八类默认21×21×11。三维新数据集名称版本为2，内部几何修订号为3。旧参数面数据与旧模型不能仅改名后复用。

| 类别 | 生成族（简称） | 输出数据集 | 默认外推评估参数 |
| --- | --- | --- | --- |
| 3D-1 | `cylindrical_domain_with_airfoil_hole` (`cah_t`) | `cah_t_ext_2_500_21_21_11` | `0.06 0.38` |
| 3D-2 | `sidewall_deformed_block` (`swb_a`) | `swb_a_ext_2_500_21_21_11` | `0.02 0.34` |
| 3D-3 | `rounded_u_notch_extrusion` (`run_e`) | `run_e_ext_2_500_41_41_11` | `1.52 1.55` |
| 3D-4 | `tapered_elliptic_bore_block` (`teb_b`, standard rectangle) | `teb_b_ext_2_500_21_21_11` | `0.08 0.36` |
| 3D-5 | `bent_annular_duct` (`bad_a`) | `bad_a_ext_2_500_21_21_11` | `10 105` |
| 3D-6 | `lobed_annular_duct` (`lad_a`) | `lad_a_ext_2_500_21_21_11` | `0 0.20` |
| 3D-7 | `square_to_round_transition` (`srt_r`) | `srt_r_ext_2_500_21_21_11` | `0.52 0.96` |
| 3D-8 | `variable_area_s_duct` (`sad_c`) | `sad_c_ext_2_500_21_21_11` | `0.50 1.42` |
| 3D-9 | `twisted_blade_passage` (`tbp_a`) | `tbp_a_ext_2_500_21_21_11` | `0 100` |

先逐类生成训练数据：

三维九类边界的外推与内插完整命令（包括参数范围、样本数、边界面采样与求解器参数）见 [data/3D/generation_commands.md](data/3D/generation_commands.md)。

再用三种显式网格规模完成训练和评估：

```powershell
python scripts/train/train_evaluate.py --dimension 3 --dataset cah_t_ext_2_500_21_21_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x11 --values 0.06 0.38 --seed 20261030 --run-tag v1
python scripts/train/train_evaluate.py --dimension 3 --dataset swb_a_ext_2_500_21_21_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x11 --values 0.02 0.34 --seed 20261030 --run-tag v1
python scripts/train/train_evaluate.py --dimension 3 --dataset run_e_ext_2_500_41_41_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x11 --values 1.52 1.55 --seed 20261030 --run-tag v1
python scripts/train/train_evaluate.py --dimension 3 --dataset teb_b_ext_2_500_21_21_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x11 --values 0.08 0.36 --seed 20261030 --run-tag v1
python scripts/train/train_evaluate.py --dimension 3 --dataset bad_a_ext_2_500_21_21_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x11 --values 10 105 --seed 20261030 --run-tag v1
python scripts/train/train_evaluate.py --dimension 3 --dataset lad_a_ext_2_500_21_21_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x11 --values 0 0.20 --seed 20261030 --run-tag v1
python scripts/train/train_evaluate.py --dimension 3 --dataset srt_r_ext_2_500_21_21_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x11 --values 0.52 0.96 --seed 20261030 --run-tag v1
python scripts/train/train_evaluate.py --dimension 3 --dataset sad_c_ext_2_500_21_21_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x11 --values 0.50 1.42 --seed 20261030 --run-tag v1
python scripts/train/train_evaluate.py --dimension 3 --dataset tbp_a_ext_2_500_21_21_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x11 --values 0 100 --seed 20261030 --run-tag v1
```

三维 `41x121x41` 的 PDE 基线和质量分析成本显著更高；显存或时长受限时，可先只保留前两个网格形状，后续只运行评估阶段补充最大规模。两维生成器都会拒绝同名目录；三维确需替换时才显式追加 `--force`。数据目录被 `.gitignore` 忽略，需要在本地生成或单独获取。

### 单独训练、单独评估与输出

完整流程和所有可选参数见 [二维正式训练—评估命令](configs/2D/train_evaluate_cli.md) 与 [三维正式训练—评估命令](configs/3D/train_evaluate_cli.md)。只训练可使用其中链接的 `train_cli.md`；只评估已有 checkpoint 时，优先在统一入口追加 `--evaluate-only`，它会固定选择 `best.pt` 或 `--checkpoint final` 的 `final.pt`，而不重新训练。`--evaluate-only --gpus 0 1 2` 会把各模型的单方法评估并行分派到指定 GPU，随后只做一次联合对比汇总。评估结果位于 `outputs/results/`，包含数值质量汇总、非法单元统计、边界误差、生成时间和网格对比图；默认不生成逐指标质量图，可追加 `--quality-plots all` 或指标名。默认同时保留普通网格对比图和红色精确连续边界叠加图；使用 `--no-exact-boundary-overlay` 可只保留普通图。三维精确边界仅叠加于所选中截面图。

重新执行正式训练时会复用配置一致且已完整保存的模型。三个模型均齐备后，缺失的训练对比图和汇总会自动补到 `outputs/training/<维度>/<数据集>/comparison/`。三维评估默认计算体网格和 `z_mid`；仅评估 `z_mid` 用 `--no-volume-evaluation --middle-sections z_mid`，仅体网格用 `--middle-sections none`，全部开启用 `--volume-evaluation --middle-sections all`。已有匹配网格或缓存时，可使用 `--evaluate-only --analysis-only` 只补图片和分析，不重新推理或求解 PDE；该模式不能与 `--force` 同时使用。

只生成训练对比图时，可运行 `python scripts/train/compare_training.py --dimension 3 --dataset swb_a_ext_2_500_21_21_11`，二维使用 `--dimension 2`。

Linux 多 GPU 动态队列（每张卡完成当前模型的训练和评估后领取下一项）：

```bash
python scripts/train/train_evaluate.py --dimension 2 --dataset-mode both --models hmonet meshonet pimeshonet --grids 101 201 401 --seed 20261030 --run-tag v1 --gpus 0 1 2 3
```

## 消融实验

二维消融实验通过统一入口运行。当前八套实验依次研究硬边界约束、采样策略、网络架构、PDE求导方式、屏障超参数、有限差分步长、坐标增强和损失项组成：

```bash
python scripts/experiments/run.py ex1 baseline rational_affine coons_residual --dataset run_ext_1_500_41_41 --seed 20261030 --grids 101 201 401 --tag v1
```

消融实验也可使用同一动态队列：

```bash
python scripts/experiments/run.py ex2 baseline all --dataset air_t_ext_1_500_41_41 --seed 20261030 --grids 101 201 401 --tag v1 --gpus 0 1 2 3
```

消融入口中 `--gpus` 与串行模式的 `--device` 互斥；正式入口会用队列分配自动替换 `--train-args`/`--evaluate-args` 中的设备值。正式训练的队列粒度为“数据集 × 模型”：同一数据集选择多个模型时，这些模型可在不同 GPU 并行完成训练及各自的单模型评估（同时保存该模型、TFI、PDE 的网格和质量结果）。全部模型任务结束后，入口把这些可复用结果写入标准 `outputs/results/` 缓存；联合阶段只读取已生成网格，计算跨方法质量汇总和对比图，不会再次执行网络推理、TFI 或 PDE 求解。队列通过 `CUDA_VISIBLE_DEVICES` 隔离物理 GPU，每个子进程内部统一使用 `cuda:0`。每张 GPU 在本次队列内同时只运行一个任务；任务结束后立即领取下一项，不检查也不等待队列之外的 GPU 进程。日志和队列状态分别写入 `outputs/gpu_queue/` 或对应实验的 `queue_logs/`。

多 GPU 子进程不写入批次级进度条；每轮训练结束的总损失、各激活损失分量及固定网格 MSE 均保存在任务日志中，主终端默认每500轮转发一次完整汇总。使用 `--queue-terminal-interval 1000` 可改为每1000轮显示一次。单 GPU 命令不受此设置影响，仍保留原有终端输出。

每个队列任务是完整的“训练→评估”单元，只有两阶段都完成才释放该 GPU；消融命令显式使用 `--train-only` 或 `--evaluate-only` 时则只执行所选阶段。`Ctrl+C`/`SIGTERM` 会取消整个队列及所有当前子任务，不受 `--continue-on-error` 影响。

不同实验套件及参数见 [experiments/README.md](experiments/README.md)。实验配置与结果位于 `experiments/ex*/`。

## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

测试采用 pytest。仅安装 `requirements.txt` 可运行项目，但不包含完整测试所需的 pytest。

## 引用、贡献与许可

学术工作中使用本项目时，请优先使用 [CITATION.cff](CITATION.cff) 中的软件引用元数据。当前为未正式打标签的开发版；论文录用、真实作者姓名、ORCID 和 DOI 将在正式发布前更新。

- 贡献流程与科研变更要求：[CONTRIBUTING.md](CONTRIBUTING.md)
- 版本变更记录：[CHANGELOG.md](CHANGELOG.md)
- 开源许可：[BSD 3-Clause License](LICENSE)
- 自动化检查：[GitHub Actions CI](.github/workflows/ci.yml)

## 补充文档

- [完整文档索引](docs/README.md)
- [环境、跨平台运行与代码同步](docs/environment.md)
- [命令与多 GPU 队列速查](docs/command_reference.md)
- [参数化数据集、边界族与曲线接口](docs/dataset_generation.md)
- [评估体系、指标与输出结构](docs/evaluation_system.md)
- [实验复现、指纹与 checkpoint 兼容性](docs/reproducibility.md)
- [已知限制与结果解释](docs/known_limitations.md)
- [二维 Poisson 下游案例](docs/downstream_poisson_2d.md)
- [二维训练命令](configs/2D/train_cli.md)
- [二维评估命令](configs/2D/evaluate_cli.md)
- [三维训练命令](configs/3D/train_cli.md)
- [三维评估命令](configs/3D/evaluate_cli.md)
- [二维训练与评估联合入口](configs/2D/train_evaluate_cli.md)
- [三维训练与评估联合入口](configs/3D/train_evaluate_cli.md)
- [消融实验](experiments/README.md)
- [HMONet 训练架构](HMONET_TRAINING_ARCHITECTURE.md)
- 本机已生成二维数据集目录：`data/raw/2D/`（生成后会出现本地目录说明）
- 本机已生成三维数据集目录：`data/raw/3D/`（生成后会出现本地目录说明）

run_e默认规模按要求改为41×41×11。实测槽深0.55、1.50的端面及体网格通过；1.52存在16个翻转体单元，1.55的两个端面各有2个内凹四边形。此设置不表示所有评估参数均合法，既有端面合法性拦截仍生效；61×21×11的通过记录保留在核验报告中。
