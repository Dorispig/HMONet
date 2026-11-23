# 2D 九类外推正式实验：临时命令清单

在项目根目录运行，先生成数据集，再训练三个模型、补全训练对比图并进行五方法评估。默认不进行外部边界后处理。

九类新数据集默认41×41。已有air_t六类核验中的翼型结果来自21×21训练集，保留不改名；若复用该模型，命令中的数据集应使用`air_t_ext_1_500_21_21`。

生成命令见[data/2D/generation_commands.md](../../data/2D/generation_commands.md)。

```powershell
# circle_airfoil_t (air_t)
python scripts/train/train_evaluate.py --dimension 2 --dataset air_t_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.62 0.65 --seed 20261030 --run-tag v1

# eccentric_semiannulus (esa)
python scripts/train/train_evaluate.py --dimension 2 --dataset esa_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.72 0.75 --seed 20261030 --run-tag v1

# rect_top_semiellipse (rse)
python scripts/train/train_evaluate.py --dimension 2 --dataset rse_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.42 0.46 0.5 0.55 0.6 --seed 20261030 --run-tag v1

# rounded_u_notch_compact (run)
python scripts/train/train_evaluate.py --dimension 2 --dataset run_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 1.52 1.55 --seed 20261030 --run-tag v1

# smooth_constricted_channel (scc)
python scripts/train/train_evaluate.py --dimension 2 --dataset scc_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.42 0.45 0.47 --seed 20261030 --run-tag v1

# variable_width_s_channel (vsc)
python scripts/train/train_evaluate.py --dimension 2 --dataset vsc_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.85 0.9 0.95 --seed 20261030 --run-tag v1

# symmetric_convex_tri (tri_p)
python scripts/train/train_evaluate.py --dimension 2 --dataset tri_p_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.42 0.48 --seed 20261030 --run-tag v1

# gear_spline (gsp_a)
python scripts/train/train_evaluate.py --dimension 2 --dataset gsp_a_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.19 0.2 --seed 20261030 --run-tag v1

# four_convex_arcs (fca_a)
python scripts/train/train_evaluate.py --dimension 2 --dataset fca_a_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 125 130 --seed 20261030 --run-tag v1

```

仅评估：在上述命令追加`--evaluate-only`；多GPU：追加`--gpus 0 1 2`。只训练：使用`python scripts/train/train_2d.py --dataset <数据集名称> --model hmonet,meshonet,pimeshonet --seed 20261030`。`--check-only`仅核验配置，不是仅训练。全部三个模型训练完或跳过后会检查并补充训练对比图。

质量图片默认不生成；`--quality-plots all`生成完整质量图，也可指定指标。普通网格图及精确边界叠加图默认同时生成；`--no-exact-boundary-overlay`关闭叠加图。其余选项见本目录train_cli.md、evaluate_cli.md和train_evaluate_cli.md。
