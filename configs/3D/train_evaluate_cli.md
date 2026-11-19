# 三维正式训练—评估命令行

统一入口为 `scripts/train/train_evaluate.py`。三维的核心参数与二维一致；`--grids` 接受一个或多个三维网格形状。

## 基本规则

| 参数 | 三维含义 |
|---|---|
| `--dataset` | 直接写 `swb_a_ext_1_500_11_21_11`，不需要 `data/raw/3D/` 前缀；相对路径和绝对路径仍兼容。 |
| `--models` | 空格分隔模型名，如 `hmonet meshonet pimeshonet`。 |
| `--grids` | 评估网格形状列表。推荐显式写 `11x21x11 21x41x21`；纯数字 `21 31` 才自动展开为 `21×21×21`、`31×31×31`。 |
| `--values` | 空格分隔物理参数值，如 `0.02 0.34`。 |
| `--run-tag` | 可读的运行标记，建议写 `v1`。 |

模型为 `hmonet`、`meshonet`、`pimeshonet`、`qghmonet`；`both`、`pi`、`all` 的含义与二维一致。

## 六类规范外推流程

三维评估配置已登记六类规范外推数据集：

| 数据集 | 测试参数 |
| --- | --- |
| `cah_t_ext_1_500_11_21_11` | `0.06 0.38` |
| `swb_a_ext_1_500_11_21_11` | `0.02 0.34` |
| `run_e_ext_1_500_41_41_11` | `1.52 1.55` |
| `teb_b_ext_1_500_11_21_11` | `0.08 0.36` |
| `bad_a_ext_1_500_11_21_11` | `10 105` |
| `lad_a_ext_1_500_11_21_11` | `0 0.20` |

从无数据开始的 12 条完整生成命令统一维护在 [data/3D/generation_commands.md](../../data/3D/generation_commands.md)。`--dataset-mode both` 只会读取 `configs/3D/train.yaml` 中声明的一对默认数据集；要复现全部六类外推集，请按该文件中的六条显式 `--dataset` 命令分别运行。

六类外推集可直接复制执行的正式训练—评估命令另见 [formal_external_commands.md](formal_external_commands.md)。

## 一键运行正式对比

~~~powershell
python scripts/train/train_evaluate.py --dimension 3 --dataset swb_a_ext_1_500_11_21_11 --models hmonet meshonet pimeshonet --grids 11x21x11 21x41x21 --values 0.02 0.34 --seed 20261030 --run-tag v1
~~~

## 仅运行一个模型或预检查

~~~powershell
# HMONet
python scripts/train/train_evaluate.py --dimension 3 --dataset swb_a_ext_1_500_11_21_11 --models hmonet --grids 11x21x11 21x41x21 --values 0.02 0.34 --seed 20261030 --run-tag v1

# 仅检查数据、网格和评估 case
python scripts/train/train_evaluate.py --dimension 3 --dataset swb_a_ext_1_500_11_21_11 --models hmonet meshonet pimeshonet --grids 11x21x11 21x41x21 --values 0.02 0.34 --seed 20261030 --run-tag v1 --check-only
~~~

## 仅评估已有 checkpoint

`--evaluate-only` 不训练模型，而是自动定位所选数据集、模型和硬约束方式对应的既有 checkpoint，并显式固定使用 `best.pt`（或 `--checkpoint final` 的 `final.pt`）。该模式会保留完整网格与质量产物，便于之后直接复用。

~~~powershell
# 重新评估三个既有正式模型及 TFI/PDE；不重新训练
python scripts/train/train_evaluate.py --dimension 3 --dataset swb_a_ext_1_500_11_21_11 --models hmonet meshonet pimeshonet --evaluate-only --grids 11x21x11 21x41x21 41x81x41 --values 0.02 0.34 --seed 20261030 --run-tag v1

# 使用既有 final.pt，而不是 best.pt
python scripts/train/train_evaluate.py --dimension 3 --dataset swb_a_ext_1_500_11_21_11 --models hmonet meshonet pimeshonet --evaluate-only --checkpoint final --grids 11x21x11 21x41x21 41x81x41 --values 0.02 0.34 --seed 20261030 --run-tag v1
~~~

若不存在所需模型的兼容 checkpoint，命令会明确报错，不会静默改用其他模型或 checkpoint。评估可使用单卡 `--device cuda:0`，也可附加 `--gpus 0 1 2`：后者会将各模型的既有 checkpoint 分配到不同 GPU 生成网格与单方法质量结果，全部完成后仅执行一次联合对比汇总，不会重新训练。

```powershell
python scripts/train/train_evaluate.py --dimension 3 --dataset cah_t_ext_1_500_11_21_11 --models hmonet meshonet pimeshonet --evaluate-only --grids 11x21x11 21x41x21 41x81x41 --values 0.06 0.38 --seed 20261030 --run-tag v1 --gpus 0 1 2
```

## 常用覆盖

| 需求 | 追加参数 |
|---|---|
| 修改最大轮数 | `--epochs 50000` |
| 修改批大小 | `--batch-size 8` |
| 修改初始学习率 | `--lr 0.0004` |
| 指定设备 | `--device cuda:0` |
| 改用最终模型评估 | `--checkpoint final` |
| 强制重新评估 | `--force` |
| 强制重新训练 | `--force-train` |
| 生成指定质量图 | `--quality-plots minimum_scaled_jacobian center_condition_number` |
| 选择三维中截面 | `--middle-sections z_mid`，也支持 `x_mid y_mid z_mid` 的组合、`all`（默认）和 `none` |

例如使用已有模型，评估体网格及论文常用的 `z_mid` 中截面。

~~~powershell
python scripts/train/train_evaluate.py --dimension 3 --dataset swb_a_ext_1_500_11_21_11 --models hmonet meshonet pimeshonet --evaluate-only --grids 11x21x11 21x41x21 --values 0.02 0.34 --middle-sections z_mid
~~~

截面的计算方式及汇总字段见 [三维评估说明](evaluate_cli.md)。

三维推荐按几何纵横比显式写每个网格：`--grids 11x31x11 21x61x21` 分别表示两个 `(n_x, n_y, n_z)` 网格。纯数字写法 `--grids 21 31` 是尺度扫描，自动生成两个立方网格 `21×21×21`、`31×31×31`；不建议把二维常用的 401 直接用于三维。旧式逗号写法 `--grid-shapes 11x31x11,21x61x21` 在此正式入口中也兼容，但不得与 `--grids` 同时使用。

## 训练阶段自动对比

当同一数据集的 HMONet、MeshONet 与 PI-MeshONet 都具有完整训练记录时，训练结束后会在 `outputs/training/3D/<dataset>/comparison/` 自动生成：`mse_comparison.png`、相对 MSE 收敛图、训练时长/轮次图及 CSV/JSON 汇总。训练记录未变且这些文件齐全时会直接复用；若图或汇总缺失，下一次非强制训练命令会补齐，不会因补图重训已完成模型。

## 多 GPU 队列

~~~bash
python scripts/train/train_evaluate.py \
  --dimension 3 \
  --dataset-mode both \
  --models hmonet meshonet pimeshonet \
  --grids 11x31x11 21x61x21 \
  --seed 20261030 \
  --run-tag v1 \
  --gpus 0 1 2 3
~~~

队列按“数据集 × 模型”分配任务。比如一个数据集同时选择 HMONet、MeshONet、PI-MeshONet，会形成三个独立的“训练 → 单模型评估”任务，可同时占用三张 GPU；每个任务训练后立即保存自身以及 TFI、PDE 的完整网格、质量和计时结果。全部模型任务结束后，这些生成缓存会被汇入标准 `outputs/results/3D/`；联合阶段直接读取已保存网格，只计算跨方法质量汇总和对比图，不会重复网络推理、TFI 或 PDE 求解。除非显式传入 `--force-regenerate` 或 `--force`。

## 正式流程与输出

1. 解析短数据集名并检查三维评估 case；
2. 读取训练配置及命令行覆盖；
3. 训练选定模型，保存 best/final checkpoint；
4. 使用本次 checkpoint 在每个 `--grids × --values` 组合上评估；
5. 保存 TFI、PDE 与模型的体网格/切片质量、边界误差、推理时间和可视化结果。

~~~text
outputs/
├── training/3D/
├── checkpoints/3D/
├── tensorboard/3D/
├── results/3D/
└── gpu_queue/formal/        # 仅多 GPU 队列
~~~

只重新评估已有模型时，使用 [三维评估命令](evaluate_cli.md)。只训练时，使用 [三维训练命令](train_cli.md)。
