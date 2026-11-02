# HMONet

HMONet 是面向二维和三维结构化网格生成的研究项目。项目使用参数曲线和参数曲面描述精确边界，提供数据集生成、模型训练、网格评估和消融实验流程。

当前实现包括：

- HMONet、MeshONet、PI-MeshONet、PI-HMONet 等模型；
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

## 数据集

数据集默认保存在：

```text
data/raw/2D/<dataset_name>/
data/raw/3D/<dataset_name>/
```

二维数据集目前包括圆形外边界与 NACA 翼型孔、偏心半环、半椭圆边界、三角凸起、光滑收缩通道、S 形通道、波纹样条边界和圆角 U 形槽等。三维数据集目前主要包括操场形柱体与圆柱孔边界。数据集仍在持续补充。

生成二维数据集：

```bash
python src/data/generation/make_dataset_parametric_2d.py --family circle_airfoil_t --split interpolation --size 1000 --grid-shape 21x20
```

生成三维数据集：

```bash
python src/data/generation/make_dataset_parametric_3d.py --family sch_r --split interpolation --size 1000 --grid-shape 11x31x11
```

生成器不会覆盖同名数据集。数据目录被 `.gitignore` 忽略，需要在本地生成或单独获取。

## 训练

二维训练：

```bash
python scripts/train/train_2d.py --config configs/2D/train.yaml --model both --dataset-mode both
```

三维训练：

```bash
python scripts/train/train_3d.py --config configs/3D/train.yaml --model both --dataset-mode both
```

`--model both` 训练 HMONet 和 MeshONet；其他模型及覆盖参数可通过 `--help` 查看。训练记录和模型默认保存在 `outputs/training/` 与 `outputs/checkpoints/`。

## 评估

二维评估示例：

```bash
python scripts/evaluate/evaluate_2d.py --config configs/2D/evaluate.yaml --case air_t_ext_0_1000_21_20 --methods tfi,pde,meshonet,hmonet --grid-shapes 51x50
```

三维评估示例：

```bash
python scripts/evaluate/evaluate_3d.py --config configs/3D/evaluate.yaml --case sch_r_ext_0_1000_11_31_11 --methods tfi,pde,meshonet,hmonet --grid-shapes 21x21x21
```

评估结果默认保存在 `outputs/results/`，包括质量指标、非法单元统计、边界误差、生成时间和对比图。默认紧凑模式会在汇总完成后删除网格数组；下游求解或二次分析需要网格时，应使用 `--generation-only` 或 `--no-compact-outputs`。详见[评估体系与输出结构](docs/evaluation_system.md)。

## 训练并评估

```bash
python scripts/train/train_evaluate.py --dimension 2 --train-config configs/2D/train.yaml --evaluate-config configs/2D/evaluate.yaml --evaluate-methods tfi,pde,meshonet,hmonet --train-args "--model both --dataset-mode both"
```

将 `--dimension 2` 改为 `--dimension 3`，并使用三维配置文件即可运行三维流程。

Linux 多 GPU 动态队列（每个模型—数据集训练占用一张卡，完成后自动领取下一项）：

```bash
python scripts/train/train_evaluate.py --dimension 2 --train-config configs/2D/train.yaml --evaluate-config configs/2D/evaluate.yaml --evaluate-methods tfi,pde,meshonet,pimeshonet,hmonet --train-args "--model hmonet,meshonet,pimeshonet --dataset-mode both" --gpus 0 1 2 3
```

## 消融实验

二维消融实验通过精简入口运行：

```bash
python scripts/experiments/run.py ex3 jacobian_interpolation --dataset air_t_ext_0_1000_41_20 --seed 20260802 20260917 20261031 --grids 101 201 401 --tag paper_v1
```

消融实验也可使用同一动态队列：

```bash
python scripts/experiments/run.py ex10 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --tag physics_v1 --gpus 0 1 2 3
```

消融入口中 `--gpus` 与串行模式的 `--device` 互斥；正式入口会用队列分配自动替换 `--train-args`/`--evaluate-args` 中的设备值。队列通过 `CUDA_VISIBLE_DEVICES` 隔离物理 GPU，每个子进程内部统一使用 `cuda:0`。每张 GPU 在本次队列内同时只运行一个任务；任务结束后立即领取下一项，不检查也不等待队列之外的 GPU 进程。日志和队列状态分别写入 `outputs/gpu_queue/` 或对应实验的 `queue_logs/`。

多 GPU 子进程不写入批次级进度条；每轮训练结束的总损失、各激活损失分量及固定网格 MSE 均保存在任务日志中，主终端默认每500轮转发一次完整汇总。使用 `--queue-terminal-interval 1000` 可改为每1000轮显示一次。单 GPU 命令不受此设置影响，仍保留原有终端输出。

每个队列任务是完整的“训练→评估”单元，只有两阶段都完成才释放该 GPU；消融命令显式使用 `--train-only` 或 `--evaluate-only` 时则只执行所选阶段。`Ctrl+C`/`SIGTERM` 会取消整个队列及所有当前子任务，不受 `--continue-on-error` 影响。

不同实验套件及参数见 [experiments/README.md](experiments/README.md)。实验配置与结果位于 `experiments/ex*/`。

## 测试

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests -p "test_*.py" -v
python -m pytest tests -q
```

项目同时包含 `unittest` 类与 pytest 风格测试函数，两条测试命令均应执行。仅安装 `requirements.txt` 可运行项目，但不包含完整测试所需的 pytest。

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
- [训练与评估联合入口](scripts/train/train_evaluate_cli.md)
- [消融实验](experiments/README.md)
- [HMONet 训练架构](HMONET_TRAINING_ARCHITECTURE.md)
- [二维已生成数据集](data/raw/2D/README.md)
- [三维已生成数据集](data/raw/3D/README.md)
