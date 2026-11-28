# 2D 九类外推正式实验：临时命令清单

在项目根目录运行，先生成数据集，再训练三个模型、补全训练对比图并进行五方法评估。默认不进行外部边界后处理。以下逐项示例保留三个默认模型，文件末尾提供含 ResMetaMesh 的四模型全数据集批量命令。

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

## 训练 ResMetaMesh 并生成六方法对比

一次处理九个数据集可运行下方批量入口。单个阶段失败后继续其他数据集，最后统一汇总。每次运行的阶段日志和 history.jsonl 位于 outputs/reports/2D/resmetamesh_batch/。

```powershell
python scripts/experiments/run_resmetamesh_batch.py --device cuda:0
```

在项目根目录运行。每条命令仅训练所选数据集的 ResMetaMesh，默认每批 64 个几何，500 条数据训练 625 轮、执行 5000 次 Adam 更新。评估时再使用 HMONet、MeshONet 和 PI-MeshONet 的已有检查点，与 TFI、PDE-Winslow 一起生成六方法对比。ResMetaMesh 默认读取 `final.pt`，其余模型默认读取 `best.pt`。外部边界后处理沿用正式评估默认设置。

```powershell
# circle_airfoil_t (air_t)
python scripts/train/train_evaluate.py --dimension 2 --dataset air_t_ext_1_500_41_41 --models resmetamesh --evaluate-methods tfi,pde,meshonet,pimeshonet,hmonet,resmetamesh --grids 101 201 401 --values 0.62 0.65 --seed 20261030 --run-tag v1

# eccentric_semiannulus (esa)
python scripts/train/train_evaluate.py --dimension 2 --dataset esa_ext_1_500_41_41 --models resmetamesh --evaluate-methods tfi,pde,meshonet,pimeshonet,hmonet,resmetamesh --grids 101 201 401 --values 0.72 0.75 --seed 20261030 --run-tag v1

# rect_top_semiellipse (rse)
python scripts/train/train_evaluate.py --dimension 2 --dataset rse_ext_1_500_41_41 --models resmetamesh --evaluate-methods tfi,pde,meshonet,pimeshonet,hmonet,resmetamesh --grids 101 201 401 --values 0.42 0.46 0.5 0.55 0.6 --seed 20261030 --run-tag v1

# rounded_u_notch_compact (run)
python scripts/train/train_evaluate.py --dimension 2 --dataset run_ext_1_500_41_41 --models resmetamesh --evaluate-methods tfi,pde,meshonet,pimeshonet,hmonet,resmetamesh --grids 101 201 401 --values 1.52 1.55 --seed 20261030 --run-tag v1

# smooth_constricted_channel (scc)
python scripts/train/train_evaluate.py --dimension 2 --dataset scc_ext_1_500_41_41 --models resmetamesh --evaluate-methods tfi,pde,meshonet,pimeshonet,hmonet,resmetamesh --grids 101 201 401 --values 0.42 0.45 0.47 --seed 20261030 --run-tag v1

# variable_width_s_channel (vsc)
python scripts/train/train_evaluate.py --dimension 2 --dataset vsc_ext_1_500_41_41 --models resmetamesh --evaluate-methods tfi,pde,meshonet,pimeshonet,hmonet,resmetamesh --grids 101 201 401 --values 0.85 0.9 0.95 --seed 20261030 --run-tag v1

# symmetric_convex_tri (tri_p)
python scripts/train/train_evaluate.py --dimension 2 --dataset tri_p_ext_1_500_41_41 --models resmetamesh --evaluate-methods tfi,pde,meshonet,pimeshonet,hmonet,resmetamesh --grids 101 201 401 --values 0.42 0.48 --seed 20261030 --run-tag v1

# gear_spline (gsp_a)
python scripts/train/train_evaluate.py --dimension 2 --dataset gsp_a_ext_1_500_41_41 --models resmetamesh --evaluate-methods tfi,pde,meshonet,pimeshonet,hmonet,resmetamesh --grids 101 201 401 --values 0.19 0.2 --seed 20261030 --run-tag v1

# four_convex_arcs (fca_a)
python scripts/train/train_evaluate.py --dimension 2 --dataset fca_a_ext_1_500_41_41 --models resmetamesh --evaluate-methods tfi,pde,meshonet,pimeshonet,hmonet,resmetamesh --grids 101 201 401 --values 125 130 --seed 20261030 --run-tag v1
```

## 一次性训练并评估全部九个数据集

数据集生成后从项目根目录运行。该命令按 `evaluate.yaml` 的 `evaluation.default_cases` 读取九个外推数据集，并使用各自配置的网格规模和评估参数。每个数据集运行 HMONet、MeshONet、PI-MeshONet、ResMetaMesh，再与 TFI、PDE-Winslow 生成六方法对比。

```bash
python scripts/train/run_formal_batch.py --dimension 2
```

可追加 `--gpus 0 1 2 3`，每张卡完成一个数据集的全部训练和评估后领取下一项。单卡可指定 `--device cuda:0`。默认种子为 `20261030`，run-tag 为 `v1`。失败的数据集会记录并跳过，全部任务结束后汇总。加 `--dry-run` 可查看命令和调度计划。匹配配置的已完成训练会复用。
