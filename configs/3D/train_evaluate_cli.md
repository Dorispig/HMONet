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
| `ugc_d_ext_1_500_11_21_11` | `0.20 1.45` |
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

## 常用覆盖

| 需求 | 追加参数 |
|---|---|
| 修改最大轮数 | `--epochs 50000` |
| 修改批大小 | `--batch-size 8` |
| 修改初始学习率 | `--lr 0.0004` |
| 指定设备 | `--device cuda:0` |
| 改用最终模型评估 | `--checkpoint final` |
| 强制重新评估 | `--force` |
| 生成指定质量图 | `--quality-plots minimum_scaled_jacobian center_condition_number` |

三维推荐按几何纵横比显式写每个网格：`--grids 11x31x11 21x61x21` 分别表示两个 `(n_x, n_y, n_z)` 网格。纯数字写法 `--grids 21 31` 是尺度扫描，自动生成两个立方网格 `21×21×21`、`31×31×31`；不建议把二维常用的 401 直接用于三维。旧式逗号写法 `--grid-shapes 11x31x11,21x61x21` 在此正式入口中也兼容，但不得与 `--grids` 同时使用。

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

队列按“数据集 × 模型”分配任务。比如一个数据集同时选择 HMONet、MeshONet、PI-MeshONet，会形成三个独立的“训练 → 单模型评估”任务，可同时占用三张 GPU；每个任务训练后立即完成其独立评估。全部模型任务结束后，入口会在一张指定 GPU 上补做一次正式的多模型汇总对比评估，写入标准 `outputs/results/3D/` 目录。

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
