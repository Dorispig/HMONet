# 命令与多 GPU 队列速查

所有命令均从项目根目录执行。参数的最终事实来源是对应入口的 `--help` 和 YAML；本文只列稳定组合。

## 正式训练与评估

| 目标 | 命令 |
| --- | --- |
| 二维训练 | `python scripts/train/train_2d.py --config configs/2D/train.yaml` |
| 三维训练 | `python scripts/train/train_3d.py --config configs/3D/train.yaml` |
| 二维评估 | `python scripts/evaluate/evaluate_2d.py --config configs/2D/evaluate.yaml --case <case>` |
| 三维评估 | `python scripts/evaluate/evaluate_3d.py --config configs/3D/evaluate.yaml --case <case>` |
| 二维训练并评估 | `python scripts/train/train_evaluate.py --dimension 2 --dataset <数据集名称> --models hmonet meshonet pimeshonet --grids 101 201 401 --values <参数1> <参数2> --run-tag v1` |
| 三维训练并评估 | `python scripts/train/train_evaluate.py --dimension 3 --dataset <数据集名称> --models hmonet meshonet pimeshonet --grids 11x31x11 21x61x21 --values <参数1> <参数2> --run-tag v1` |

三维 `--grids` 的每一项可显式写成 `n_x × n_y × n_z`，例如 `11x31x11`；纯数字 `21 31` 才表示立方网格 `21×21×21`、`31×31×31`。正式入口也兼容旧式 `--grid-shapes 11x31x11,21x61x21`，但两种写法不可混用。

常用覆盖：

```bash
python scripts/train/train_2d.py --model hmonet,meshonet,pimeshonet --dataset air_t_ext_1_500_41_41 --dataset-mode extrapolation --seed 20261030 --device cuda
python scripts/evaluate/evaluate_2d.py --case run_ext_1_500_41_41 --values 1.52 1.55 --grids 101 201 401 --methods tfi,pde,meshonet,pimeshonet,hmonet
```

## 消融实验

```bash
# 单个实验
python scripts/experiments/run.py ex1 rational_affine --dataset run_ext_1_500_41_41 --seed 20261030 --grids 101 201 401 --tag v1

# 基线和套件内全部实验
python scripts/experiments/run.py ex2 baseline all --dataset air_t_ext_1_500_41_41 --seed 20261030 --grids 101 201 401 --tag v1

# 只评估且不生成图片
python scripts/experiments/run.py ex1 rational_affine --evaluate-only --grids 101 201 --no-plots
```

`all` 不包含基线，只有显式写出 `baseline` 才运行基线。`--force` 会替换所选任务的既有运行；使用前应先执行 `--dry-run`。`--train-only` 与 `--evaluate-only` 互斥。

## 多 GPU 动态队列

正式流程和二维消融套件共用动态队列。每个“模型或实验变体 × 数据集 × 随机种子”任务绑定一张指定 GPU；默认完整执行“训练→评估”后才释放该 GPU。

```bash
# 消融实验
python scripts/experiments/run.py ex2 baseline all --dataset air_t_ext_1_500_41_41 --seed 20261030 --grids 101 201 401 --tag v1 --gpus 0 1 2 3

# 正式实验
python scripts/train/train_evaluate.py --dimension 2 --dataset-mode both --models hmonet meshonet pimeshonet --grids 101 201 401 --seed 20261030 --run-tag v1 --gpus 0 1 2 3
```

调度规则：

- “空闲”表示该 GPU 在本次队列内没有任务，不检查队列外进程；
- 子进程内部统一看到 `cuda:0`；
- `--train-only`/`--evaluate-only` 可将任务改为单阶段；
- `Ctrl+C`/`SIGTERM` 终止整个队列和所有子任务；
- `--continue-on-error` 允许普通失败释放 GPU 并继续，但最终仍汇总失败。

多 GPU 消融日志按 `experiments/<suite>/queue_logs/<dataset>/<experiment>/seed_<seed>/<run-tag>/` 存放；正式实验采用相同顺序，位于 `outputs/gpu_queue/formal/<dataset>/<experiment>/seed_<seed>/<run-tag>/`。训练指纹只保存在队列元数据中，不再出现在路径里。训练和评估属于同一个原子任务时写入 `train_evaluate.log`；单阶段任务分别写入 `training.log` 或 `evaluation.log`。正式队列会保留每个模型的完整单模型评估，并把方法级网格与质量缓存汇入 `outputs/results/2D|3D/`；随后联合阶段只读取这些结果以生成跨方法比较，不重复模型推理、TFI 或 PDE 求解。再次运行同一目录对应的任务时会先清除旧日志，再覆盖写入，避免新旧输出混合。正式流程使用 `--run-tag` 指定标签；标签仅限字母、数字、`.`、`_`、`-`。路径检查仅针对即将创建的真实目录或文件；超过预警阈值时提示，达到当前系统的真实路径上限才会报错。`--queue-terminal-interval 1000` 表示主终端每 1000 epoch 转发一次完整汇总；任务日志仍保存每个完整 epoch 的结果，单 GPU 输出不受影响。

常见故障：训练身份不匹配时应换 tag 或确认后使用 `--force`；任务瞬间退出应先查看对应日志的首个 traceback；CUDA OOM 时减小 batch/query batch，并避免两个队列同时使用同一物理卡。

## 数据、测试和可视化

```bash
python src/data/generation/make_dataset_parametric_2d.py --family circle_airfoil_t --split extrapolation --version 1 --size 500 --grid-shape 21x21 --parameter-range 0.055 0.6
python src/data/generation/make_dataset_parametric_3d.py --family swb_a --split extrapolation --version 2 --size 500 --grid-shape 11x21x11
python -m unittest discover -s tests -p "test_*.py" -v
python -m pytest tests -q
tensorboard --logdir outputs/training --port 6006
```

数据生成见[参数化数据集](dataset_generation.md)，评估产物见[评估体系](evaluation_system.md)。


三维数据生成默认使用CPU，也可通过 `--solver-backend cuda --gpus 0 1` 启用数据集级多GPU生成；完整参数和示例见[三维生成说明](../data/3D/generation_commands.md)。训练GPU参数与数据生成GPU参数分别设置。

三维 GPU 数据生成采用可选 cuDSS float64 稀疏直接求解，与 CPU 共用矩阵组装和外层迭代。cuDSS 已纳入总依赖，统一安装 `python -m pip install -r requirements.txt`；CPU 仍为默认，运行时不调用 cuDSS。完整 CPU/多 GPU 命令见 [生成命令](../data/3D/generation_commands.md)。原 `--linear-tolerance`、`--linear-max-iter` 已移除。

GPU 以完整数据集分配任务；详细进度按前几个样本耗时和每 10% 完成度写入 `outputs/gpu_queue/generation/3D/<数据集名称>/generation.log`。可用 `--queue-log-root` 修改日志根目录，终端仅显示任务概况和错误。

### 三维数据集 GPU 队列

```bash
python scripts/data/generate_datasets.py --dimension 3 --families all --split ext --size 500 --version 2 --solver-backend cuda --gpus 0 1 2 3 --timing-samples 3 --queue-log-root outputs/gpu_queue/generation
```

一张 GPU 处理一个完整数据集，四张 GPU 最多同时处理四个；数据集内部样本依次生成。单类生成只需 `--gpus 0`。完整的参数范围及 CPU/单 GPU/多 GPU 示例统一见[三维生成命令](../data/3D/generation_commands.md)。

## ResMetaMesh 与全数据集批量入口

| 任务 | 命令 |
|---|---|
| 二维九个默认外推数据集 | `python scripts/train/run_formal_batch.py --dimension 2` |
| 三维九个默认外推数据集，同一队列四卡执行 | `python scripts/train/run_formal_batch.py --dimension 3 --gpus 0 1 2 3` |

批量入口通过 `formal-four` 选择四个模型，逐数据集生成六方法结果。训练和联合评估属于同一个数据集任务。任务失败先记日志再继续。多 GPU 日志位于 `outputs/gpu_queue/formal/batch/<维度>/seed_<种子>/<run-tag>/<数据集>/run.log`，并在批次目录汇总为 `queue_summary.json`。追加 `--dry-run` 只查看调度。ResMetaMesh 单独训练或评估与其他模型共用正式入口，模型名为 `resmetamesh`，默认评估检查点为 `final.pt`。
