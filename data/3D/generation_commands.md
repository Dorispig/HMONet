# 3D 九类数据集生成命令（内插与外推）

在项目根目录运行，九类边界的外推、内插各提供 CPU 和多 GPU 两种完整命令，共36条单类命令，另附批量命令。命令显式给出样本数、节点规模、版本、参数范围及求解设置。

九类三维边界的两个端面分别采用201×201节点二维Winslow求解，再以精确Coons边界加三次样条位移定义连续参数面。四侧面在原参数域使用TFI，保留原解析几何；不做参数重分布、不复用上下端面求值。体内仍求解三维PDE。run_e默认41×41×11，其余八类默认21×21×11。三维新数据集名称版本为2，内部几何修订号为3。旧参数面数据与旧模型不能仅改名后复用。

## 九类当前默认参数

样本数500，版本2，CPU workers=4；除run_e为41×41×11外均为21×21×11。表中的“几何函数默认值”用于单个边界示例；生成训练数据时会在指定区间均匀取样，并非只生成这个默认值。

| 边界公式 | 可变参数 | 几何函数默认值 | 外推训练范围 | 内插训练范围 | 节点规模 |
|---|---|---:|---|---|---|
| [cah_t](boundaries/cah_t.md) | `thickness` | 0.12 | `[[0.08, 0.34]]` | `[[0.06, 0.15], [0.25, 0.35]]` | 21×21×11 |
| [swb_a](boundaries/swb_a.md) | `sidewall_amplitude` | 0.15 | `[[0.04, 0.3]]` | `[[0.03, 0.1], [0.2, 0.3]]` | 21×21×11 |
| [run_e](boundaries/run_e.md) | `depth` | 1 | `[[0.55, 1.5]]` | `[[0.55, 1.0], [1.3, 1.5]]` | 41×41×11 |
| [teb_b](boundaries/teb_b.md) | `minor_axis_out` | 0.24 | `[[0.12, 0.32]]` | `[[0.12, 0.18], [0.27, 0.32]]` | 21×21×11 |
| [bad_a](boundaries/bad_a.md) | `bend_angle_deg` | 60 | `[[25.0, 85.0]]` | `[[20.0, 45.0], [65.0, 90.0]]` | 21×21×11 |
| [lad_a](boundaries/lad_a.md) | `amplitude` | 0.1 | `[[0.04, 0.16]]` | `[[0.02, 0.08], [0.12, 0.18]]` | 21×21×11 |
| [srt_r](boundaries/srt_r.md) | `outlet_radius` | 0.75 | `[[0.55, 0.92]]` | `[[0.58, 0.68], [0.8, 0.9]]` | 21×21×11 |
| [sad_c](boundaries/sad_c.md) | `area_ratio_out` | 1 | `[[0.55, 1.35]]` | `[[0.6, 0.82], [1.08, 1.3]]` | 21×21×11 |
| [tbp_a](boundaries/tbp_a.md) | `twist_angle_deg` | 18 | `[[0.0, 90.0]]` | `[[0.0, 30.0], [60.0, 90.0]]` | 21×21×11 |

公式、固定几何参数和图片见表中的链接。以下命令是当前默认配置的显式快照；修改注册表后，批量命令读取新默认值，而显式命令仍使用写出的参数。

## 批量生成

每组 CPU / GPU 命令二选一即可，后端不同但目标数据集名称相同；完整数据集已存在时会跳过。GPU 示例使用 `0 1 2 3`，单卡可改为 `0`，两卡可改为 `0 1`。省略 `--solver-backend` 时仍默认 CPU。

```powershell
# 一次生成九类外推数据集
# CPU（默认后端）
python scripts/data/generate_datasets.py --dimension 3 --families all --split ext --size 500 --version 2 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw

# 多 GPU（按服务器实际情况修改 GPU 编号）
python scripts/data/generate_datasets.py --dimension 3 --families all --split ext --size 500 --version 2 --solver-backend cuda --gpus 0 1 2 3 --linear-tolerance 1e-12 --linear-max-iter 4000 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw

# 只生成九类内插数据集
# CPU（默认后端）
python scripts/data/generate_datasets.py --dimension 3 --families all --split int --size 500 --version 2 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw

# 多 GPU（按服务器实际情况修改 GPU 编号）
python scripts/data/generate_datasets.py --dimension 3 --families all --split int --size 500 --version 2 --solver-backend cuda --gpus 0 1 2 3 --linear-tolerance 1e-12 --linear-max-iter 4000 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw

# 同时生成内插与外推（共18个数据集）
# CPU（默认后端）
python scripts/data/generate_datasets.py --dimension 3 --families all --split both --size 500 --version 2 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw

# 多 GPU（按服务器实际情况修改 GPU 编号）
python scripts/data/generate_datasets.py --dimension 3 --families all --split both --size 500 --version 2 --solver-backend cuda --gpus 0 1 2 3 --linear-tolerance 1e-12 --linear-max-iter 4000 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw
```

将`--split ext`改为`int`只生成九类内插集，改为`both`生成18个。`--dry-run`只列出任务。`--families`可直接写边界缩写列表；`new`只是最近新增三类的快捷筛选，与其余族共用同一注册表。

参数值在给定区间内均匀取样；无需随机种子。`--workers`是CPU并行求解进程数。已有同名完整数据集跳过。同名旧参数面数据会明确报错并保留原文件；确认替换时使用`--force`，否则使用新的输出目录。

## 九类默认外推命令（逐条运行）

```powershell
# cah_t: thickness ∈ [[0.08, 0.34]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family cah_t --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.08 0.34 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 多 GPU（按服务器实际情况修改 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family cah_t --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.08 0.34 --solver-backend cuda --gpus 0 1 2 3 --linear-tolerance 1e-12 --linear-max-iter 4000 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# swb_a: sidewall_amplitude ∈ [[0.04, 0.3]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family swb_a --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.04 0.3 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 多 GPU（按服务器实际情况修改 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family swb_a --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.04 0.3 --solver-backend cuda --gpus 0 1 2 3 --linear-tolerance 1e-12 --linear-max-iter 4000 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# run_e: depth ∈ [[0.55, 1.5]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family run_e --split extrapolation --version 2 --size 500 --grid-shape 41x41x11 --parameter-range 0.55 1.5 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 多 GPU（按服务器实际情况修改 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family run_e --split extrapolation --version 2 --size 500 --grid-shape 41x41x11 --parameter-range 0.55 1.5 --solver-backend cuda --gpus 0 1 2 3 --linear-tolerance 1e-12 --linear-max-iter 4000 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# teb_b: minor_axis_out ∈ [[0.12, 0.32]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family teb_b --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.12 0.32 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 多 GPU（按服务器实际情况修改 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family teb_b --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.12 0.32 --solver-backend cuda --gpus 0 1 2 3 --linear-tolerance 1e-12 --linear-max-iter 4000 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# bad_a: bend_angle_deg ∈ [[25.0, 85.0]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family bad_a --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 25 85 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 多 GPU（按服务器实际情况修改 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family bad_a --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 25 85 --solver-backend cuda --gpus 0 1 2 3 --linear-tolerance 1e-12 --linear-max-iter 4000 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# lad_a: amplitude ∈ [[0.04, 0.16]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family lad_a --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.04 0.16 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 多 GPU（按服务器实际情况修改 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family lad_a --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.04 0.16 --solver-backend cuda --gpus 0 1 2 3 --linear-tolerance 1e-12 --linear-max-iter 4000 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# srt_r: outlet_radius ∈ [[0.55, 0.92]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family srt_r --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.55 0.92 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 多 GPU（按服务器实际情况修改 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family srt_r --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.55 0.92 --solver-backend cuda --gpus 0 1 2 3 --linear-tolerance 1e-12 --linear-max-iter 4000 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# sad_c: area_ratio_out ∈ [[0.55, 1.35]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family sad_c --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.55 1.35 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 多 GPU（按服务器实际情况修改 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family sad_c --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.55 1.35 --solver-backend cuda --gpus 0 1 2 3 --linear-tolerance 1e-12 --linear-max-iter 4000 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# tbp_a: twist_angle_deg ∈ [[0.0, 90.0]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family tbp_a --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0 90 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 多 GPU（按服务器实际情况修改 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family tbp_a --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0 90 --solver-backend cuda --gpus 0 1 2 3 --linear-tolerance 1e-12 --linear-max-iter 4000 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

```

## 九类默认内插命令（逐条运行）

```powershell
# cah_t: thickness ∈ [[0.06, 0.15], [0.25, 0.35]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family cah_t --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.06 0.15 --parameter-range 0.25 0.35 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 多 GPU（按服务器实际情况修改 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family cah_t --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.06 0.15 --parameter-range 0.25 0.35 --solver-backend cuda --gpus 0 1 2 3 --linear-tolerance 1e-12 --linear-max-iter 4000 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# swb_a: sidewall_amplitude ∈ [[0.03, 0.1], [0.2, 0.3]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family swb_a --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.03 0.1 --parameter-range 0.2 0.3 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 多 GPU（按服务器实际情况修改 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family swb_a --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.03 0.1 --parameter-range 0.2 0.3 --solver-backend cuda --gpus 0 1 2 3 --linear-tolerance 1e-12 --linear-max-iter 4000 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# run_e: depth ∈ [[0.55, 1.0], [1.3, 1.5]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family run_e --split interpolation --version 2 --size 500 --grid-shape 41x41x11 --parameter-range 0.55 1 --parameter-range 1.3 1.5 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 多 GPU（按服务器实际情况修改 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family run_e --split interpolation --version 2 --size 500 --grid-shape 41x41x11 --parameter-range 0.55 1 --parameter-range 1.3 1.5 --solver-backend cuda --gpus 0 1 2 3 --linear-tolerance 1e-12 --linear-max-iter 4000 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# teb_b: minor_axis_out ∈ [[0.12, 0.18], [0.27, 0.32]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family teb_b --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.12 0.18 --parameter-range 0.27 0.32 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 多 GPU（按服务器实际情况修改 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family teb_b --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.12 0.18 --parameter-range 0.27 0.32 --solver-backend cuda --gpus 0 1 2 3 --linear-tolerance 1e-12 --linear-max-iter 4000 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# bad_a: bend_angle_deg ∈ [[20.0, 45.0], [65.0, 90.0]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family bad_a --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 20 45 --parameter-range 65 90 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 多 GPU（按服务器实际情况修改 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family bad_a --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 20 45 --parameter-range 65 90 --solver-backend cuda --gpus 0 1 2 3 --linear-tolerance 1e-12 --linear-max-iter 4000 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# lad_a: amplitude ∈ [[0.02, 0.08], [0.12, 0.18]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family lad_a --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.02 0.08 --parameter-range 0.12 0.18 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 多 GPU（按服务器实际情况修改 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family lad_a --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.02 0.08 --parameter-range 0.12 0.18 --solver-backend cuda --gpus 0 1 2 3 --linear-tolerance 1e-12 --linear-max-iter 4000 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# srt_r: outlet_radius ∈ [[0.58, 0.68], [0.8, 0.9]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family srt_r --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.58 0.68 --parameter-range 0.8 0.9 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 多 GPU（按服务器实际情况修改 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family srt_r --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.58 0.68 --parameter-range 0.8 0.9 --solver-backend cuda --gpus 0 1 2 3 --linear-tolerance 1e-12 --linear-max-iter 4000 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# sad_c: area_ratio_out ∈ [[0.6, 0.82], [1.08, 1.3]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family sad_c --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.6 0.82 --parameter-range 1.08 1.3 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 多 GPU（按服务器实际情况修改 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family sad_c --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.6 0.82 --parameter-range 1.08 1.3 --solver-backend cuda --gpus 0 1 2 3 --linear-tolerance 1e-12 --linear-max-iter 4000 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# tbp_a: twist_angle_deg ∈ [[0.0, 30.0], [60.0, 90.0]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family tbp_a --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0 30 --parameter-range 60 90 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 多 GPU（按服务器实际情况修改 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family tbp_a --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0 30 --parameter-range 60 90 --solver-backend cuda --gpus 0 1 2 3 --linear-tolerance 1e-12 --linear-max-iter 4000 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

```


## 参数在哪里修改

`DATASET_FAMILIES`保存几何固定参数、内插/外推范围、默认网格规模；`SELECTED_DATASETS`决定无参数运行的任务、版本及显式网格规模。修改规模时同时检查这两个变量。`DEFAULT_GENERATION_OPTIONS`保存样本数、workers和体PDE求解设置。三维端面精度由每族`fixed_parameters`中的`cap_nx/cap_ny/cap_max_iter/cap_tolerance`控制，与体网格和网络输入规模分开。

run_e默认规模按要求改为41×41×11。实测槽深0.55、1.50的端面及体网格通过；评估参数1.52、1.55的PDE体网格分别存在16、20个翻转单元，1.55的两个端面各有2个内凹四边形。数据集生成会拦截非法端面；评估会正常记录网格质量，不能把求解收敛或评估结束当作网格合法。61×21×11的通过记录保留在核验报告中。

## 常用参数速查

| 参数 | 含义 |
|---|---|
| `--family` | 单类生成入口使用边界缩写；批量入口使用 `--families`。 |
| `--split` | `ext/extrapolation`为外推训练集，`int/interpolation`为内插训练集；批量入口还支持`both`。 |
| `--size` | 几何样本数，默认500。 |
| `--version` | 数据集名称中的版本号，默认2。 |
| `--grid-shape` | 每个方向的节点数，不是单元数。 |
| `--parameter-range MIN MAX` | 可变几何参数的区间；重复填写可指定不连续区间。 |
| `--workers` | 独立样本的CPU并行求解进程数，默认4。 |
| `--winslow-max-iter` | 体/平面PDE迭代上限，默认1000。 |
| `--winslow-tolerance` | PDE求解停止容差，默认1e-6。 |
| `--timing-samples` | 输出详细生成时间的前几个样本，默认3。 |
| `--output-dir` / `--output-root` | 二维单类入口用前者；三维和批量入口用后者。 |
| `--dry-run` | 批量入口仅列出任务，不生成数据。 |

训练、评估及GPU选项见[正式实验命令](../../configs/3D/formal_external_commands.md)。

## 三维 CPU / 多 GPU 数据集生成

默认仍使用原 CPU 求解器，`--workers 4` 表示四个 CPU 工作进程。
显式指定 `--solver-backend cuda` 才启用 GPU；不指定 `--gpus` 时使用当前可见的 GPU 0。
GPU 模式下每张卡一个独立进程、同时求解一个几何样本；完成后领取下一样本。
九类数据集依次生成，每个数据集内部样本动态并行。`--workers` 不控制 GPU 并发数。

完整 CPU / GPU 命令已在上方“批量生成”“九类默认外推命令”“九类默认内插命令”逐组列出。

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--solver-backend` | `cpu` | `cpu` 保留原 SciPy 求解；`cuda` 使用 Torch GPU 求解。 |
| `--gpus` | CUDA 模式下为 `0` | 空格分隔的 GPU 编号；按 `CUDA_VISIBLE_DEVICES` 后的可见编号解释。必须搭配 `--solver-backend cuda`。 |
| `--linear-tolerance` | `1e-12` | GPU 内层线性系统的相对残差容差，与外层网格变化停止阈值不同。 |
| `--linear-max-iter` | `4000` | 每次线性系统求解的迭代上限。 |
| `--workers` | `4` | 仅控制 CPU 模式的样本并行数。 |

GPU 使用 float64，保留端面9点、体网格19点的原 Winslow 离散格式和混合导数项。
上下端面各自求解，仍使用相同的 Coons＋三次样条位移参数面，侧面几何、数据格式和训练读取方式不变。
线性求解采用 Jacobi 预条件 BiCGSTAB，同时核对真实线性残差；未收敛时明确报错，不静默写入成功标签。
CPU/GPU 的舍入误差可能不同，不承诺逐位相等；GPU 后端和求解诊断会保存到元数据。
GPU 不自动判断整张卡是否空闲，用户应指定允许使用的卡。Ctrl+C 会终止本次生成队列的全部子进程。
各 GPU 不直接写最终数据文件；父进程按原样本编号统一保存，避免并行冲突。
边界面非法仍中断；体网格首尾参数的非法单元按原规则在保存完整数据集后警告。
已有完整数据集仍按原规则跳过；切换后端不会自动覆盖它，需要重新生成时显式使用 `--force`。
需安装支持 CUDA 的 PyTorch；无 CUDA 环境时 CPU 流程仍可用。GPU 加速幅度取决于网格规模及显卡双精度性能。
