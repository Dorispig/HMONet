# 3D 九类外推正式实验：临时命令清单

在项目根目录运行，先生成数据集，再训练三个模型、补全训练对比图并进行五方法评估。默认不进行外部边界后处理。

当前九类默认评估节点规模统一为 `41x41x11` 和 `81x81x11`；不填写 `--grids` 时读取 case 的 `grid_shapes`。固定 z 方向为11点适合横截面加密比较，不代表三方向同步加密的三维收敛研究。训练网格改变后，边界输入维度也改变，需生成新规模数据并重新训练，不能重命名旧 checkpoint。

补充核验：run_e在81×81×11下，1.52与1.55的端面及PDE体网格均通过本次局部合法性检查，翻转单元数均为0；41×41×11的已知凹单元仍如实保留。详见[规模和耗时报告](../../reports/validation/pde_caps_3d/grid_scale_review.md)。

九类三维边界的两个端面分别采用201×201节点二维Winslow求解，再以精确Coons边界加三次样条位移定义连续参数面。四侧面在原参数域使用TFI，保留原解析几何；不做参数重分布、不复用上下端面求值。体内仍求解三维PDE。run_e默认41×41×11，其余八类默认21×21×11。三维新数据集名称版本为2，内部几何修订号为3。旧参数面数据与旧模型不能仅改名后复用。

默认评估体网格和z_mid；仅截面追加`--no-volume-evaluation --middle-sections z_mid`，全部截面追加`--middle-sections all`。run_e示例采用当前默认41×41×11；该规模在深凹评估参数下有已知非法单元，详见下方说明。

生成命令见[data/3D/generation_commands.md](../../data/3D/generation_commands.md)。

```powershell
# cylindrical_domain_with_airfoil_hole (cah_t)
python scripts/train/train_evaluate.py --dimension 3 --dataset cah_t_ext_2_500_21_21_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x11 --values 0.06 0.38 --seed 20261030 --run-tag v1

# sidewall_deformed_block (swb_a)
python scripts/train/train_evaluate.py --dimension 3 --dataset swb_a_ext_2_500_21_21_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x11 --values 0.02 0.34 --seed 20261030 --run-tag v1

# rounded_u_notch_extrusion (run_e)
python scripts/train/train_evaluate.py --dimension 3 --dataset run_e_ext_2_500_41_41_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x11 --values 1.52 1.55 --seed 20261030 --run-tag v1

# tapered_elliptic_bore_block (teb_b)
python scripts/train/train_evaluate.py --dimension 3 --dataset teb_b_ext_2_500_21_21_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x11 --values 0.08 0.36 --seed 20261030 --run-tag v1

# bent_annular_duct (bad_a)
python scripts/train/train_evaluate.py --dimension 3 --dataset bad_a_ext_2_500_21_21_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x11 --values 10 105 --seed 20261030 --run-tag v1

# lobed_annular_duct (lad_a)
python scripts/train/train_evaluate.py --dimension 3 --dataset lad_a_ext_2_500_21_21_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x11 --values 0 0.2 --seed 20261030 --run-tag v1

# square_to_round_transition (srt_r)
python scripts/train/train_evaluate.py --dimension 3 --dataset srt_r_ext_2_500_21_21_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x11 --values 0.52 0.96 --seed 20261030 --run-tag v1

# variable_area_s_duct (sad_c)
python scripts/train/train_evaluate.py --dimension 3 --dataset sad_c_ext_2_500_21_21_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x11 --values 0.5 1.42 --seed 20261030 --run-tag v1

# twisted_blade_passage (tbp_a)
python scripts/train/train_evaluate.py --dimension 3 --dataset tbp_a_ext_2_500_21_21_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x11 --values 0 100 --seed 20261030 --run-tag v1

```

仅评估：在上述命令追加`--evaluate-only`；多GPU：追加`--gpus 0 1 2`。只训练：使用`python scripts/train/train_3d.py --dataset <数据集名称> --model hmonet,meshonet,pimeshonet --seed 20261030`。`--check-only`仅核验配置，不是仅训练。全部三个模型训练完或跳过后会检查并补充训练对比图。

质量图片默认不生成；`--quality-plots all`生成完整质量图，也可指定指标。普通网格图及精确边界叠加图默认同时生成；`--no-exact-boundary-overlay`关闭叠加图。其余选项见本目录train_cli.md、evaluate_cli.md和train_evaluate_cli.md。

新训练记录会保存参数面方案及几何修订号；同名旧参数面模型不会自动复用，新参数面需要重新训练。旧模型文件保留。

run_e默认规模按要求改为41×41×11。实测槽深0.55、1.50的端面及体网格通过；评估参数1.52、1.55的PDE体网格分别存在16、20个翻转单元，1.55的两个端面各有2个内凹四边形。数据集生成会拦截非法端面；评估会正常记录网格质量，不能把求解收敛或评估结束当作网格合法。61×21×11的通过记录保留在核验报告中。

三维数据生成默认使用CPU，也可通过 `--solver-backend cuda --gpus 0 1` 启用样本级多GPU生成；完整参数和示例见[三维生成说明](../../data/3D/generation_commands.md)。训练GPU参数与数据生成GPU参数分别设置。

三维 GPU 数据生成采用可选 cuDSS float64 稀疏直接求解，与 CPU 共用矩阵组装和外层迭代。首次使用安装 `python -m pip install -r requirements-gpu-generation.txt`；CPU 仍为默认，不需要此依赖。完整 CPU/多 GPU 命令见 [生成命令](../../data/3D/generation_commands.md)。原 `--linear-tolerance`、`--linear-max-iter` 已移除。
