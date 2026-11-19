# 二维正式训练命令行

二维单独训练入口为 `scripts/train/train_2d.py`。日常正式运行建议使用 `scripts/train/train_evaluate.py`，使训练和评估始终绑定同一次 checkpoint；完整串联流程见 [二维训练—评估说明](train_evaluate_cli.md)。

## 短数据集名

`--dataset` 可写：

1. `air_t_ext_1_500_41_41`；
2. `data/raw/2D/air_t_ext_1_500_41_41`；
3. 绝对路径。

短名称会自动在 `data/raw/2D` 内解析。训练网格由数据集元数据决定。

## 推荐的训练—评估命令

~~~powershell
python scripts/train/train_evaluate.py --dimension 2 --dataset air_t_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.62 0.65 --seed 20261030 --run-tag v1
~~~

## 仅训练

配置一致且已有完整训练记录和 best/final checkpoint 的模型会直接复用。三个正式模型齐备后，如果比较图或汇总缺失，命令会在 `outputs/training/2D/<dataset>/comparison/` 补齐。需要重训时显式加 `--force-train`。

~~~powershell
python scripts/train/train_2d.py --dataset air_t_ext_1_500_41_41 --dataset-mode extrapolation --model hmonet --seed 20261030

python scripts/train/train_2d.py --dataset air_t_ext_1_500_41_41 --dataset-mode extrapolation --model hmonet,meshonet,pimeshonet --seed 20261030

python scripts/train/train_2d.py --dataset-mode both --model hmonet,meshonet,pimeshonet --seed 20261030
~~~

## 常用覆盖

| 参数 | 示例 | 含义 |
|---|---|---|
| `--model` | `hmonet` | 选择模型，多个模型使用逗号分隔。 |
| `--dataset` | `air_t_ext_1_500_41_41` | 训练数据集。 |
| `--dataset-mode` | `extrapolation` | 未指定数据集时选择 interpolation、extrapolation 或 both。 |
| `--epochs` | `50000` | 最大训练轮数。 |
| `--batch-size` | `32` | 训练批大小。 |
| `--lr` | `0.0004` | 初始学习率。 |
| `--seed` | `20261030` | 随机种子。 |
| `--device` | `cuda:0` | 训练设备。 |

## 只生成训练对比图

~~~bash
python scripts/train/compare_training.py --dimension 2 --dataset air_t_ext_1_500_41_41 run_ext_1_500_41_41
~~~

可指定一个或多个数据集。脚本读取三个正式模型各自最新的已完成训练记录，将对比图和汇总保存到 `outputs/training/2D/<dataset>/comparison/`。训练日志存放在其他位置时，可用 `--training-root` 指定包含 `2D/` 或 `3D/` 的目录。

模型选择为 `hmonet`、`meshonet`、`pimeshonet`、`qghmonet`；`both`、`pi`、`all` 分别表示 HMONet+MeshONet、PI-MeshONet、全部四个模型。

完整参数：

~~~powershell
python scripts/train/train_2d.py --help
~~~
