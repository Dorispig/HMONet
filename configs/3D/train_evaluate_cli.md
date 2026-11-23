# 三维正式训练—评估命令行

统一入口为 `scripts/train/train_evaluate.py`。三维的核心参数与二维一致；`--grids` 接受一个或多个三维网格形状。

九类三维边界的两个端面分别采用201×201节点二维Winslow求解，再以精确Coons边界加三次样条位移定义连续参数面。四侧面在原参数域使用TFI，保留原解析几何；不做参数重分布、不复用上下端面求值。体内仍求解三维PDE。run_e默认41×41×11，其余八类默认21×21×11。所有新数据集名称版本为1，内部几何修订号为3。旧参数面数据与旧模型不能仅改名后复用。

## 基本规则

| 参数 | 三维含义 |
|---|---|
| `--dataset` | 直接写 `swb_a_ext_1_500_21_21_11`，不需要 `data/raw/3D/` 前缀；相对路径和绝对路径仍兼容。 |
| `--models` | 空格分隔模型名，如 `hmonet meshonet pimeshonet`。 |
| `--grids` | 评估网格形状列表。当前默认 `41x41x11 81x81x11`；纯数字 `21 31` 才自动展开为 `21×21×21`、`31×31×31`。 |
| `--values` | 空格分隔物理参数值，如 `0.02 0.34`。 |
| `--run-tag` | 可读的运行标记，建议写 `v1`。 |

模型为 `hmonet`、`meshonet`、`pimeshonet`、`qghmonet`；`both`、`pi`、`all` 的含义与二维一致。

## 九类规范外推流程

三维评估配置已登记九类规范外推数据集：

| 数据集 | 测试参数 |
| --- | --- |
| `cah_t_ext_1_500_21_21_11` | `0.06 0.38` |
| `swb_a_ext_1_500_21_21_11` | `0.02 0.34` |
| `run_e_ext_1_500_41_41_11` | `1.52 1.55` |
| `teb_b_ext_1_500_21_21_11` | `0.08 0.36` |
| `bad_a_ext_1_500_21_21_11` | `10 105` |
| `lad_a_ext_1_500_21_21_11` | `0 0.20` |
| `srt_r_ext_1_500_21_21_11` | `0.52 0.96` |
| `sad_c_ext_1_500_21_21_11` | `0.50 1.42` |
| `tbp_a_ext_1_500_21_21_11` | `0 100` |

从无数据开始的 18 条完整生成命令统一维护在 [data/3D/generation_commands.md](../../data/3D/generation_commands.md)。`--dataset-mode both` 只会读取 `configs/3D/train.yaml` 中声明的一对默认数据集；要复现全部九类外推集，请按该文件中的九条显式 `--dataset` 命令分别运行。

九类外推集可直接复制执行的正式训练—评估命令另见 [formal_external_commands.md](formal_external_commands.md)。

## 一键运行正式对比

~~~powershell
python scripts/train/train_evaluate.py --dimension 3 --dataset swb_a_ext_1_500_21_21_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x11 --values 0.02 0.34 --seed 20261030 --run-tag v1
~~~

## 仅运行一个模型或预检查

~~~powershell
# HMONet
python scripts/train/train_evaluate.py --dimension 3 --dataset swb_a_ext_1_500_21_21_11 --models hmonet --grids 41x41x11 81x81x11 --values 0.02 0.34 --seed 20261030 --run-tag v1

# 仅检查数据、网格和评估 case
python scripts/train/train_evaluate.py --dimension 3 --dataset swb_a_ext_1_500_21_21_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x11 --values 0.02 0.34 --seed 20261030 --run-tag v1 --check-only
~~~

## 仅评估已有 checkpoint

`--evaluate-only` 不训练模型，而是自动定位所选数据集、模型和硬约束方式对应的既有 checkpoint，并显式固定使用 `best.pt`（或 `--checkpoint final` 的 `final.pt`）。该模式会保留完整网格与质量产物，便于之后直接复用。

若完整网格或匹配缓存已经存在，只想补充截面图片或重新整理分析结果，可在 `--evaluate-only` 后直接添加 `--analysis-only`。该模式不会重新推理网络或求解 PDE，且不能与 `--force` 同时使用。

~~~powershell
# 重新评估三个既有正式模型及 TFI/PDE；不重新训练
python scripts/train/train_evaluate.py --dimension 3 --dataset swb_a_ext_1_500_21_21_11 --models hmonet meshonet pimeshonet --evaluate-only --grids 41x41x11 81x81x11 --values 0.02 0.34 --seed 20261030 --run-tag v1

# 使用既有 final.pt，而不是 best.pt
python scripts/train/train_evaluate.py --dimension 3 --dataset swb_a_ext_1_500_21_21_11 --models hmonet meshonet pimeshonet --evaluate-only --checkpoint final --grids 41x41x11 81x81x11 --values 0.02 0.34 --seed 20261030 --run-tag v1
~~~

若不存在所需模型的兼容 checkpoint，命令会明确报错，不会静默改用其他模型或 checkpoint。评估可使用单卡 `--device cuda:0`，也可附加 `--gpus 0 1 2`：后者会将各模型的既有 checkpoint 分配到不同 GPU 生成网格与单方法质量结果，全部完成后仅执行一次联合对比汇总，不会重新训练。

```powershell
python scripts/train/train_evaluate.py --dimension 3 --dataset cah_t_ext_1_500_21_21_11 --models hmonet meshonet pimeshonet --evaluate-only --grids 41x41x11 81x81x11 --values 0.06 0.38 --seed 20261030 --run-tag v1 --gpus 0 1 2
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
| 体网格评估 | 默认 `--volume-evaluation`；用 `--no-volume-evaluation` 跳过六面体质量及体网格图，仍保存完整坐标网格供截面提取及复用 |
| 选择三维中截面 | 默认 `--middle-sections z_mid`，支持 `x_mid y_mid z_mid` 的任意组合、`all` 和 `none`；独立于体网格开关 |
| 精确边界叠加图 | 默认开启；`--no-exact-boundary-overlay` 关闭。仅在选中的中截面生成额外的 `middle_section_mesh_with_exact_boundary.png`，不修改体网格图。 |
| 仅补分析图片 | `--evaluate-only --analysis-only`；要求已有匹配的 `mesh.npy` 或生成缓存，不要添加 `--force`。 |

例如使用已有模型，评估体网格及论文常用的 `z_mid` 中截面。

~~~powershell
python scripts/train/train_evaluate.py --dimension 3 --dataset swb_a_ext_1_500_21_21_11 --models hmonet meshonet pimeshonet --evaluate-only --grids 41x41x11 81x81x11 --values 0.02 0.34 --middle-sections z_mid
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

run_e默认规模按要求改为41×41×11。实测槽深0.55、1.50的端面及体网格通过；评估参数1.52、1.55的PDE体网格分别存在16、20个翻转单元，1.55的两个端面各有2个内凹四边形。数据集生成会拦截非法端面；评估会正常记录网格质量，不能把求解收敛或评估结束当作网格合法。61×21×11的通过记录保留在核验报告中。
