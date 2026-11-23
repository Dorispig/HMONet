# 三维正式训练命令行

三维单独训练入口为 `scripts/train/train_3d.py`。日常正式运行建议使用 `scripts/train/train_evaluate.py`；完整串联流程见 [三维训练—评估说明](train_evaluate_cli.md)。

## 短数据集名

`--dataset` 可直接写 `swb_a_ext_2_500_21_21_11`，无需再书写 `data/raw/3D/`。项目相对路径和绝对路径仍兼容；短名称自动在 `data/raw/3D` 中解析。训练网格由数据集元数据决定。

## 推荐的训练—评估命令

~~~powershell
python scripts/train/train_evaluate.py --dimension 3 --dataset swb_a_ext_2_500_21_21_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x11 --values 0.02 0.34 --seed 20261030 --run-tag v1
~~~

`--grids` 的每个显式形状均按 `n_x×n_y×n_z` 书写，例如 `11x31x11 21x61x21`。纯数字 `--grids 21 31` 才表示两个立方网格 `21×21×21`、`31×31×31`；旧式 `--grid-shapes 11x31x11,21x61x21` 也兼容。

## 仅训练

配置一致且已有完整训练记录和 best/final checkpoint 的模型会直接复用。三个正式模型齐备后，如果比较图或汇总缺失，命令会在 `outputs/training/3D/<dataset>/comparison/` 补齐。需要重训时显式加 `--force-train`。

~~~powershell
python scripts/train/train_3d.py --dataset swb_a_ext_2_500_21_21_11 --dataset-mode extrapolation --model hmonet --seed 20261030

python scripts/train/train_3d.py --dataset swb_a_ext_2_500_21_21_11 --dataset-mode extrapolation --model hmonet,meshonet,pimeshonet --seed 20261030

python scripts/train/train_3d.py --dataset-mode both --model hmonet,meshonet,pimeshonet --seed 20261030
~~~

| 参数 | 示例 | 含义 |
|---|---|---|
| `--model` | `hmonet` | 选择模型，多个模型使用逗号分隔。 |
| `--dataset` | `swb_a_ext_2_500_21_21_11` | 三维数据集短名称。 |
| `--dataset-mode` | `extrapolation` | 未指定数据集时选择训练划分。 |
| `--epochs` | `50000` | 最大训练轮数。 |
| `--batch-size` | `8` | 训练批大小。 |
| `--lr` | `0.0004` | 初始学习率。 |
| `--seed` | `20261030` | 随机种子。 |
| `--device` | `cuda:0` | 训练设备。 |

## 只生成训练对比图

~~~bash
python scripts/train/compare_training.py --dimension 3 --dataset swb_a_ext_2_500_21_21_11 cah_t_ext_2_500_21_21_11
~~~

每个数据集使用三个正式模型各自最新的已完成训练记录，输出全程 MSE、相对 MSE、训练成本图和汇总表。输出位于 `outputs/training/3D/<dataset>/comparison/`，来源记录见 `training_comparison.json`。可只指定一个数据集，也可连续列出多个。训练日志存放在其他位置时，可用 `--training-root` 指定包含 `2D/` 或 `3D/` 的目录。

模型和别名与二维完全相同：`hmonet`、`meshonet`、`pimeshonet`、`qghmonet`；`both`、`pi`、`all`。

~~~powershell
python scripts/train/train_3d.py --help
~~~

新训练记录会保存参数面方案及几何修订号；同名旧参数面模型不会自动复用，新参数面需要重新训练。旧模型文件保留。
