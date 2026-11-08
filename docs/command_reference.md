# 命令与多 GPU 队列速查

所有命令均从项目根目录执行。参数的最终事实来源是对应入口的 `--help` 和 YAML；本文只列稳定组合。

## 正式训练与评估

| 目标 | 命令 |
| --- | --- |
| 二维训练 | `python scripts/train/train_2d.py --config configs/2D/train.yaml` |
| 三维训练 | `python scripts/train/train_3d.py --config configs/3D/train.yaml` |
| 二维评估 | `python scripts/evaluate/evaluate_2d.py --config configs/2D/evaluate.yaml --case <case>` |
| 三维评估 | `python scripts/evaluate/evaluate_3d.py --config configs/3D/evaluate.yaml --case <case>` |
| 二维训练并评估 | `python scripts/train/train_evaluate.py --dimension 2 --train-config configs/2D/train.yaml --evaluate-config configs/2D/evaluate.yaml` |
| 三维训练并评估 | `python scripts/train/train_evaluate.py --dimension 3 --train-config configs/3D/train.yaml --evaluate-config configs/3D/evaluate.yaml` |

常用覆盖：

```bash
python scripts/train/train_2d.py --model hmonet,meshonet,pimeshonet --dataset-mode both --seed 20261030 --device cuda
python scripts/evaluate/evaluate_2d.py --case run_ext_0_1000_61_21 --values 1.52,1.55 --grid-shapes 101x101,201x201,401x401 --methods tfi,pde,meshonet,pimeshonet,hmonet
```

## 消融实验

```bash
# 单个实验
python scripts/experiments/run.py ex1 rational_affine --dataset run_ext_0_1000_61_21 --seed 20261030 --grids 101 201 401 --tag physics_v1

# 基线和套件内全部实验
python scripts/experiments/run.py ex2 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --tag physics_v1

# 只评估且不生成图片
python scripts/experiments/run.py ex1 rational_affine --evaluate-only --grids 101 201 --no-plots
```

`all` 不包含基线，只有显式写出 `baseline` 才运行基线。`--force` 会替换所选任务的既有运行；使用前应先执行 `--dry-run`。`--train-only` 与 `--evaluate-only` 互斥。

## 多 GPU 动态队列

正式流程和二维消融套件共用动态队列。每个“模型或实验变体 × 数据集 × 随机种子”任务绑定一张指定 GPU；默认完整执行“训练→评估”后才释放该 GPU。

```bash
# 消融实验
python scripts/experiments/run.py ex2 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --tag physics_v1 --gpus 0 1 2 3

# 正式实验
python scripts/train/train_evaluate.py --dimension 2 --train-config configs/2D/train.yaml --evaluate-config configs/2D/evaluate.yaml --evaluate-methods tfi,pde,meshonet,pimeshonet,hmonet --train-args "--model hmonet,meshonet,pimeshonet --dataset-mode both" --gpus 0 1 2 3
```

调度规则：

- “空闲”表示该 GPU 在本次队列内没有任务，不检查队列外进程；
- 子进程内部统一看到 `cuda:0`；
- `--train-only`/`--evaluate-only` 可将任务改为单阶段；
- `Ctrl+C`/`SIGTERM` 终止整个队列和所有子任务；
- `--continue-on-error` 允许普通失败释放 GPU 并继续，但最终仍汇总失败。

多 GPU 消融日志按 `experiments/<suite>/queue_logs/<dataset>/<experiment>/seed_<seed>/<run-tag>/` 存放；正式实验采用相同顺序，位于 `outputs/gpu_queue/formal/<dataset>/<experiment>/seed_<seed>/<run-tag>/`。训练指纹只保存在队列元数据中，不再出现在路径里。训练和评估属于同一个原子任务时写入 `train_evaluate.log`；单阶段任务分别写入 `training.log` 或 `evaluation.log`。再次运行同一目录对应的任务时会先清除旧日志，再覆盖写入，避免新旧输出混合。正式多 GPU 流程可用 `--run-tag`（或 `--tag`）指定标签。`--queue-terminal-interval 1000` 表示主终端每 1000 epoch 转发一次完整汇总；任务日志仍保存每个完整 epoch 的结果，单 GPU 输出不受影响。

常见故障：训练身份不匹配时应换 tag 或确认后使用 `--force`；任务瞬间退出应先查看对应日志的首个 traceback；CUDA OOM 时减小 batch/query batch，并避免两个队列同时使用同一物理卡。

## 数据、测试和可视化

```bash
python src/data/generation/make_dataset_parametric_2d.py --family circle_airfoil_t --split extrapolation --version 1 --size 500 --grid-shape 21x21 --parameter-range 0.055 0.6
python src/data/generation/make_dataset_parametric_3d.py --family sch_r --split extrapolation --version 1 --size 500 --grid-shape 11x31x11
python -m unittest discover -s tests -p "test_*.py" -v
python -m pytest tests -q
tensorboard --logdir outputs/training --port 6006
```

数据生成见[参数化数据集](dataset_generation.md)，评估产物见[评估体系](evaluation_system.md)。

