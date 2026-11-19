# 三维 v1/500 数据集生成命令

本文件列出当前确定的六类三维参数边界的内插、外推数据集命令，共 12 条。所有命令应在项目根目录执行，并写入 `data/raw/3D/`。命令显式给出所有会影响数据定义和数值求解的参数。

三维生成器会保留已存在的同名目录并输出 `[skip]`。默认六类边界均为当前正式基准。

## 无参数批量生成

```powershell
python src/data/generation/make_dataset_parametric_3d.py
```

这会按代码中的 `SELECTED_DATASETS` 顺序生成六个正式的外推数据集，统一采用 `v1/500`；其中 `run_e` 使用与二维一致的 `41x41x11` 网格，其余为 `11x21x11`。已存在的数据集会跳过，不会被覆盖。内插数据集仍应使用下文的显式命令生成。

## 外推数据集

```powershell
# cah_t_ext_1_500_11_21_11；thickness ∈ [0.08, 0.34]
python src/data/generation/make_dataset_parametric_3d.py --family cah_t --split extrapolation --version 1 --size 500 --grid-shape 11x21x11 --parameter-range 0.08 0.34 --sample-count 400 --workers 4 --seed 20260905 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# swb_a_ext_1_500_11_21_11；sidewall_amplitude ∈ [0.04, 0.30]
python src/data/generation/make_dataset_parametric_3d.py --family swb_a --split extrapolation --version 1 --size 500 --grid-shape 11x21x11 --parameter-range 0.04 0.30 --sample-count 400 --workers 4 --seed 20260905 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# run_e_ext_1_500_41_41_11；与二维 run 截面完全相同，depth ∈ [0.55, 1.50]
python src/data/generation/make_dataset_parametric_3d.py --family run_e --split extrapolation --version 1 --size 500 --grid-shape 41x41x11 --parameter-range 0.55 1.50 --sample-count 400 --workers 4 --seed 20260905 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# teb_b_ext_1_500_11_21_11；minor_axis_out ∈ [0.12, 0.32]
python src/data/generation/make_dataset_parametric_3d.py --family teb_b --split extrapolation --version 1 --size 500 --grid-shape 11x21x11 --parameter-range 0.12 0.32 --sample-count 400 --workers 4 --seed 20260905 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# bad_a_ext_1_500_11_21_11；bend_angle_deg ∈ [25, 85]
python src/data/generation/make_dataset_parametric_3d.py --family bad_a --split extrapolation --version 1 --size 500 --grid-shape 11x21x11 --parameter-range 25 85 --sample-count 400 --workers 4 --seed 20260905 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# lad_a_ext_1_500_11_21_11；amplitude ∈ [0.04, 0.16]
python src/data/generation/make_dataset_parametric_3d.py --family lad_a --split extrapolation --version 1 --size 500 --grid-shape 11x21x11 --parameter-range 0.04 0.16 --sample-count 400 --workers 4 --seed 20260905 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D
```

## 内插数据集

内插划分由两个不相交区间组成，命令中重复的 `--parameter-range` 共同定义完整的训练参数集合。

```powershell
# cah_t_int_1_500_11_21_11；thickness ∈ [0.06, 0.15] ∪ [0.25, 0.35]
python src/data/generation/make_dataset_parametric_3d.py --family cah_t --split interpolation --version 1 --size 500 --grid-shape 11x21x11 --parameter-range 0.06 0.15 --parameter-range 0.25 0.35 --sample-count 400 --workers 4 --seed 20260905 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# swb_a_int_1_500_11_21_11；sidewall_amplitude ∈ [0.03, 0.10] ∪ [0.20, 0.30]
python src/data/generation/make_dataset_parametric_3d.py --family swb_a --split interpolation --version 1 --size 500 --grid-shape 11x21x11 --parameter-range 0.03 0.10 --parameter-range 0.20 0.30 --sample-count 400 --workers 4 --seed 20260905 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# run_e_int_1_500_41_41_11；与二维 run 截面完全相同，depth ∈ [0.55, 1.00] ∪ [1.30, 1.50]
python src/data/generation/make_dataset_parametric_3d.py --family run_e --split interpolation --version 1 --size 500 --grid-shape 41x41x11 --parameter-range 0.55 1.00 --parameter-range 1.30 1.50 --sample-count 400 --workers 4 --seed 20260905 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# teb_b_int_1_500_11_21_11；minor_axis_out ∈ [0.12, 0.18] ∪ [0.27, 0.32]
python src/data/generation/make_dataset_parametric_3d.py --family teb_b --split interpolation --version 1 --size 500 --grid-shape 11x21x11 --parameter-range 0.12 0.18 --parameter-range 0.27 0.32 --sample-count 400 --workers 4 --seed 20260905 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# bad_a_int_1_500_11_21_11；bend_angle_deg ∈ [20, 45] ∪ [65, 90]
python src/data/generation/make_dataset_parametric_3d.py --family bad_a --split interpolation --version 1 --size 500 --grid-shape 11x21x11 --parameter-range 20 45 --parameter-range 65 90 --sample-count 400 --workers 4 --seed 20260905 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# lad_a_int_1_500_11_21_11；amplitude ∈ [0.02, 0.08] ∪ [0.12, 0.18]
python src/data/generation/make_dataset_parametric_3d.py --family lad_a --split interpolation --version 1 --size 500 --grid-shape 11x21x11 --parameter-range 0.02 0.08 --parameter-range 0.12 0.18 --sample-count 400 --workers 4 --seed 20260905 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D
```

`--sample-count 400` 是每个参数面用于保存/编码的离散采样密度；它不同于 `--grid-shape` 指定的 PDE 体网格节点数。生成完成后，应先复核六面共享边、八个角点以及极端参数处的体网格合法性诊断。
