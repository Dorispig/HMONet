# 二维 v1/500 数据集生成命令

本文件列出当前确定的六类二维边界的内插、外推数据集命令，共 12 条。所有命令应在项目根目录执行；数据集写入 `data/raw/2D/`。命令显式给出了边界族、划分、版本、样本数、PDE 网格、参数范围与求解器参数，避免依赖脚本默认值。

已有的 `run_ext_1_500_41_41` 会在批量生成时跳过。早期生成的 `air_t_ext_1_500_21_21` 被保留以支持已有实验；新的统一默认网格为 `41x41`，因此会生成新的 `air_t_ext_1_500_41_41`，不会覆盖旧目录。

## 无参数批量生成

```powershell
python src/data/generation/make_dataset_parametric_2d.py
```

这会按代码中的 `SELECTED_DATASETS` 顺序生成六个正式的外推数据集，统一使用 `41x41` 网格；版本和样本数为 `v1/500`。已存在的数据集会跳过，不会被覆盖。内插数据集仍应使用下文的显式命令生成。

## 外推数据集

```powershell
# air_t_ext_1_500_41_41；t ∈ [0.055, 0.60]
python src/data/generation/make_dataset_parametric_2d.py --family circle_airfoil_t --split extrapolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.055 0.60 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --workers 4 --output-dir data/raw/2D

# esa_ext_1_500_41_41；inner_center_x ∈ [0.30, 0.70]
python src/data/generation/make_dataset_parametric_2d.py --family eccentric_semiannulus --split extrapolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.30 0.70 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --workers 4 --output-dir data/raw/2D

# rse_ext_1_500_41_41；semi_minor_axis ∈ [0.00, 0.40]
python src/data/generation/make_dataset_parametric_2d.py --family rect_top_semiellipse --split extrapolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.00 0.40 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --workers 4 --output-dir data/raw/2D

# 已生成：run_ext_1_500_41_41；depth ∈ [0.55, 1.50]
python src/data/generation/make_dataset_parametric_2d.py --family rounded_u_notch_compact --split extrapolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.55 1.50 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --workers 4 --output-dir data/raw/2D

# scc_ext_1_500_41_41；depth ∈ [0.05, 0.40]
python src/data/generation/make_dataset_parametric_2d.py --family smooth_constricted_channel --split extrapolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.05 0.40 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --workers 4 --output-dir data/raw/2D

# vsc_ext_1_500_41_41；constriction ∈ [0.00, 0.80]
python src/data/generation/make_dataset_parametric_2d.py --family variable_width_s_channel --split extrapolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.00 0.80 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --workers 4 --output-dir data/raw/2D
```

## 内插数据集

内插划分由两个不相交区间组成，因此同一条命令中重复使用 `--parameter-range`。

```powershell
# air_t_int_1_500_41_41；t ∈ [0.055, 0.25] ∪ [0.35, 0.60]
python src/data/generation/make_dataset_parametric_2d.py --family circle_airfoil_t --split interpolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.055 0.25 --parameter-range 0.35 0.60 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --workers 4 --output-dir data/raw/2D

# esa_int_1_500_41_41；inner_center_x ∈ [0.30, 0.45] ∪ [0.55, 0.70]
python src/data/generation/make_dataset_parametric_2d.py --family eccentric_semiannulus --split interpolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.30 0.45 --parameter-range 0.55 0.70 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --workers 4 --output-dir data/raw/2D

# rse_int_1_500_41_41；semi_minor_axis ∈ [0.00, 0.20] ∪ [0.30, 0.50]
python src/data/generation/make_dataset_parametric_2d.py --family rect_top_semiellipse --split interpolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.00 0.20 --parameter-range 0.30 0.50 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --workers 4 --output-dir data/raw/2D

# run_int_1_500_41_41；depth ∈ [0.55, 1.00] ∪ [1.30, 1.50]
python src/data/generation/make_dataset_parametric_2d.py --family rounded_u_notch_compact --split interpolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.55 1.00 --parameter-range 1.30 1.50 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --workers 4 --output-dir data/raw/2D

# scc_int_1_500_41_41；depth ∈ [0.05, 0.25] ∪ [0.35, 0.42]
python src/data/generation/make_dataset_parametric_2d.py --family smooth_constricted_channel --split interpolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.05 0.25 --parameter-range 0.35 0.42 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --workers 4 --output-dir data/raw/2D

# vsc_int_1_500_41_41；constriction ∈ [0.00, 0.30] ∪ [0.60, 0.85]
python src/data/generation/make_dataset_parametric_2d.py --family variable_width_s_channel --split interpolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.00 0.30 --parameter-range 0.60 0.85 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --workers 4 --output-dir data/raw/2D
```

`--workers 4` 仅表示四个 CPU PDE 求解进程；机器核心较少时可下调。生成结束后请检查每个数据集目录中的元数据和极端参数网格合法性诊断。
