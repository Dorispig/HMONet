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
├── scripts/        # 训练、评估、PDE、TFI 和可视化入口
├── src/            # 数据、模型、评估和通用模块
├── tests/          # 自动化测试
├── docs/           # 补充说明
└── requirements.txt
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

评估结果默认保存在 `outputs/results/`，包括网格、质量指标、非法单元统计、边界误差、生成时间和对比图。

## 训练并评估

```bash
python scripts/train/train_evaluate.py --dimension 2 --train-config configs/2D/train.yaml --evaluate-config configs/2D/evaluate.yaml --evaluate-methods tfi,pde,meshonet,hmonet --train-args "--model both --dataset-mode both"
```

将 `--dimension 2` 改为 `--dimension 3`，并使用三维配置文件即可运行三维流程。

## 消融实验

二维消融实验通过统一入口运行：

```bash
python scripts/experiments/run_ablations_2d.py --experiment-set ex3 --ablations jacobian_interpolation --dataset air_t_ext_0_1000_41_20 --seeds 20260802 20260917 20261031 --run-tag paper_v1 --continue-on-error
```

不同实验套件及参数见 [experiments/README.md](experiments/README.md)。实验配置与结果位于 `experiments/ex*/`。

## 测试

```bash
python -m unittest discover -s tests -v
```

## 补充文档

- [训练与评估命令](scripts/train/train_evaluate_cli.md)
- [评估体系](docs/evaluation_system.md)
- [跨平台运行](docs/cross_platform.md)
- [消融实验](experiments/README.md)
