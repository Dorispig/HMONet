# 二维正式训练—评估命令行

统一入口为 `scripts/train/train_evaluate.py`。它将训练、checkpoint 固定和评估组合为一个完整任务，推荐用于正式实验。

## 基本规则

| 参数 | 二维含义 |
|---|---|
| `--dataset` | 直接写数据集名称，如 `air_t_ext_1_500_41_41`；也兼容相对路径和绝对路径。 |
| `--models` | 空格分隔模型名，如 `hmonet meshonet pimeshonet`。 |
| `--grids` | 评估边长列表。`101 201 401` 自动展开为 `101×101`、`201×201`、`401×401`。 |
| `--values` | 空格分隔的物理参数值，如 `0.62 0.65`。 |
| `--run-tag` | 可读的运行标记，建议写 `v1`。 |

模型为 `hmonet`、`meshonet`、`pimeshonet`、`qghmonet`。别名 `both` 为 HMONet + MeshONet，`pi` 为 PI-MeshONet，`all` 为全部四个模型。

## 六类规范外推流程

六类二维外推训练集及其默认测试参数为：

| 数据集 | 测试参数 |
| --- | --- |
| `air_t_ext_1_500_41_41` | `0.62 0.65` |
| `esa_ext_1_500_41_41` | `0.72 0.75` |
| `rse_ext_1_500_41_41` | `0.42 0.46 0.50 0.55 0.60` |
| `run_ext_1_500_41_41` | `1.52 1.55` |
| `scc_ext_1_500_41_41` | `0.42 0.45 0.47` |
| `vsc_ext_1_500_41_41` | `0.85 0.90 0.95` |

从无数据开始的 12 条完整生成命令统一维护在 [data/2D/generation_commands.md](../../data/2D/generation_commands.md)；本文件仅说明二维正式入口的通用参数与运行规则。

六类外推集可直接复制执行的正式训练—评估命令另见 [formal_external_commands.md](formal_external_commands.md)。

## 一键运行正式对比

~~~powershell
python scripts/train/train_evaluate.py --dimension 2 --dataset air_t_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.62 0.65 --seed 20261030 --run-tag v1
~~~

## 仅运行一个模型或预检查

~~~powershell
# HMONet
python scripts/train/train_evaluate.py --dimension 2 --dataset air_t_ext_1_500_41_41 --models hmonet --grids 101 201 401 --values 0.62 0.65 --seed 20261030 --run-tag v1

# 仅检查数据、网格和评估 case
python scripts/train/train_evaluate.py --dimension 2 --dataset air_t_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.62 0.65 --seed 20261030 --run-tag v1 --check-only
~~~

## 常用覆盖

| 需求 | 追加参数 |
|---|---|
| 修改最大轮数 | `--epochs 50000` |
| 修改批大小 | `--batch-size 32` |
| 修改初始学习率 | `--lr 0.0004` |
| 指定设备 | `--device cuda:0` |
| 改用最终模型评估 | `--checkpoint final` |
| 强制重新评估 | `--force` |
| 生成指定质量图 | `--quality-plots scaled_jacobian max_angle` |
| 生成全部质量图 | `--quality-plots all` |

默认保存网格图和数值汇总，不生成逐质量指标图。旧式 `--grid-shapes 101x101,201x201` 和逗号参数值仍兼容。

## 多 GPU 队列

~~~bash
python scripts/train/train_evaluate.py \
  --dimension 2 \
  --dataset-mode both \
  --models hmonet meshonet pimeshonet \
  --grids 101 201 401 \
  --seed 20261030 \
  --run-tag v1 \
  --gpus 0 1 2 3
~~~

队列按“数据集 × 模型”分配任务。比如一个数据集同时选择 HMONet、MeshONet、PI-MeshONet，会形成三个独立的“训练 → 单模型评估”任务，可同时占用三张 GPU；每个任务训练后立即保存自身以及 TFI、PDE 的完整网格、质量和计时结果。全部模型任务结束后，这些生成缓存会被汇入标准 `outputs/results/2D/`；联合阶段直接读取已保存网格，只计算跨方法质量汇总和对比图，不会重复网络推理、TFI 或 PDE 求解。除非显式传入 `--force-regenerate` 或 `--force`。

## 正式流程与输出

1. 解析短数据集名并检查二维评估 case；
2. 加载 YAML 和命令行覆盖项；
3. 训练所选模型并保存 best/final checkpoint；
4. 将本次 best checkpoint 显式交给评估器；
5. 遍历全部 `--grids × --values` 组合，输出 TFI、PDE 和模型的质量、边界误差、推理时间及网格图。

~~~text
outputs/
├── training/2D/
├── checkpoints/2D/
├── tensorboard/2D/
├── results/2D/
└── gpu_queue/formal/        # 仅多 GPU 队列
~~~

只重新评估已有模型时，使用 [二维评估命令](evaluate_cli.md)。只训练时，使用 [二维训练命令](train_cli.md)。
