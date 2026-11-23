# 3D 九类数据集生成命令（内插与外推）

在项目根目录运行，九类边界的外推、内插各提供 CPU 和单 GPU 两种完整命令，共36条单类命令，另附多 GPU 数据集队列命令。命令显式给出样本数、节点规模、版本、参数范围及求解设置。

九类三维边界的两个端面分别采用201×201节点二维Winslow求解，再以精确Coons边界加三次样条位移定义连续参数面。四侧面在原参数域使用TFI，保留原解析几何；不做参数重分布、不复用上下端面求值。体内仍求解三维PDE。run_e默认41×41×11，其余八类默认21×21×11。三维新数据集名称版本为2，内部几何修订号为3。旧参数面数据与旧模型不能仅改名后复用。

统一安装依赖：`python -m pip install -r requirements.txt`。GPU 模式需要 CUDA 12 PyTorch 与可加载的 cuDSS；CPU 模式仍是默认。

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

每组 CPU / GPU 命令二选一即可，后端不同但目标数据集名称相同；完整数据集已存在时会跳过。批量 GPU 示例使用 `0 1 2 3`：最多同时生成四个完整数据集，每张卡完成当前数据集的全部样本、校验、保存和示例图后，再领取下一个数据集。单卡可改为 `0`，两卡可改为 `0 1`。省略 `--solver-backend` 时仍默认 CPU。

```powershell
# 一次生成九类外推数据集
# CPU（默认后端）
python scripts/data/generate_datasets.py --dimension 3 --families all --split ext --size 500 --version 2 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw

# 多 GPU（按服务器实际情况修改 GPU 编号）
python scripts/data/generate_datasets.py --dimension 3 --families all --split ext --size 500 --version 2 --solver-backend cuda --gpus 0 1 2 3 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw --queue-log-root outputs/gpu_queue/generation

# 只生成九类内插数据集
# CPU（默认后端）
python scripts/data/generate_datasets.py --dimension 3 --families all --split int --size 500 --version 2 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw

# 多 GPU（按服务器实际情况修改 GPU 编号）
python scripts/data/generate_datasets.py --dimension 3 --families all --split int --size 500 --version 2 --solver-backend cuda --gpus 0 1 2 3 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw --queue-log-root outputs/gpu_queue/generation

# 同时生成内插与外推（共18个数据集）
# CPU（默认后端）
python scripts/data/generate_datasets.py --dimension 3 --families all --split both --size 500 --version 2 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw

# 多 GPU（按服务器实际情况修改 GPU 编号）
python scripts/data/generate_datasets.py --dimension 3 --families all --split both --size 500 --version 2 --solver-backend cuda --gpus 0 1 2 3 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw --queue-log-root outputs/gpu_queue/generation
```

只生成部分数据集（两张 GPU 动态分配三个完整数据集）：

```powershell
python scripts/data/generate_datasets.py --dimension 3 --families cah_t swb_a run_e --split ext --size 500 --version 2 --solver-backend cuda --gpus 0 1 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw --queue-log-root outputs/gpu_queue/generation
```

运行前只查看任务清单，不生成数据：

```powershell
python scripts/data/generate_datasets.py --dimension 3 --families all --split ext --size 500 --version 2 --solver-backend cuda --gpus 0 1 2 3 --queue-log-root outputs/gpu_queue/generation --dry-run
```

将`--split ext`改为`int`只生成九类内插集，改为`both`生成18个。`--dry-run`只列出任务。`--families`可直接写边界缩写列表；`new`只是最近新增三类的快捷筛选，与其余族共用同一注册表。

参数值在给定区间内均匀取样；无需随机种子。`--workers`是CPU并行求解进程数。已有同名完整数据集跳过。同名旧参数面数据会明确报错并保留原文件；确认替换时使用`--force`，否则使用新的输出目录。

## 九类默认外推命令（逐条运行）

每条命令只有一个数据集任务，因此 GPU 示例只指定一张卡。要让多张卡同时生成不同数据集，请使用上方批量入口；不要为单类命令填写四张 GPU 并期待它们共同生成该数据集。

```powershell
# cah_t: thickness ∈ [[0.08, 0.34]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family cah_t --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.08 0.34 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 单 GPU（完整生成当前数据集；可将 0 改为指定 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family cah_t --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.08 0.34 --solver-backend cuda --gpus 0 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D --queue-log-root outputs/gpu_queue/generation

# swb_a: sidewall_amplitude ∈ [[0.04, 0.3]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family swb_a --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.04 0.3 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 单 GPU（完整生成当前数据集；可将 0 改为指定 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family swb_a --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.04 0.3 --solver-backend cuda --gpus 0 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D --queue-log-root outputs/gpu_queue/generation

# run_e: depth ∈ [[0.55, 1.5]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family run_e --split extrapolation --version 2 --size 500 --grid-shape 41x41x11 --parameter-range 0.55 1.5 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 单 GPU（完整生成当前数据集；可将 0 改为指定 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family run_e --split extrapolation --version 2 --size 500 --grid-shape 41x41x11 --parameter-range 0.55 1.5 --solver-backend cuda --gpus 0 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D --queue-log-root outputs/gpu_queue/generation

# teb_b: minor_axis_out ∈ [[0.12, 0.32]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family teb_b --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.12 0.32 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 单 GPU（完整生成当前数据集；可将 0 改为指定 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family teb_b --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.12 0.32 --solver-backend cuda --gpus 0 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D --queue-log-root outputs/gpu_queue/generation

# bad_a: bend_angle_deg ∈ [[25.0, 85.0]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family bad_a --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 25 85 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 单 GPU（完整生成当前数据集；可将 0 改为指定 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family bad_a --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 25 85 --solver-backend cuda --gpus 0 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D --queue-log-root outputs/gpu_queue/generation

# lad_a: amplitude ∈ [[0.04, 0.16]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family lad_a --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.04 0.16 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 单 GPU（完整生成当前数据集；可将 0 改为指定 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family lad_a --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.04 0.16 --solver-backend cuda --gpus 0 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D --queue-log-root outputs/gpu_queue/generation

# srt_r: outlet_radius ∈ [[0.55, 0.92]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family srt_r --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.55 0.92 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 单 GPU（完整生成当前数据集；可将 0 改为指定 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family srt_r --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.55 0.92 --solver-backend cuda --gpus 0 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D --queue-log-root outputs/gpu_queue/generation

# sad_c: area_ratio_out ∈ [[0.55, 1.35]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family sad_c --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.55 1.35 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 单 GPU（完整生成当前数据集；可将 0 改为指定 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family sad_c --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.55 1.35 --solver-backend cuda --gpus 0 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D --queue-log-root outputs/gpu_queue/generation

# tbp_a: twist_angle_deg ∈ [[0.0, 90.0]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family tbp_a --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0 90 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 单 GPU（完整生成当前数据集；可将 0 改为指定 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family tbp_a --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0 90 --solver-backend cuda --gpus 0 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D --queue-log-root outputs/gpu_queue/generation

```

## 九类默认内插命令（逐条运行）

```powershell
# cah_t: thickness ∈ [[0.06, 0.15], [0.25, 0.35]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family cah_t --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.06 0.15 --parameter-range 0.25 0.35 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 单 GPU（完整生成当前数据集；可将 0 改为指定 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family cah_t --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.06 0.15 --parameter-range 0.25 0.35 --solver-backend cuda --gpus 0 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D --queue-log-root outputs/gpu_queue/generation

# swb_a: sidewall_amplitude ∈ [[0.03, 0.1], [0.2, 0.3]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family swb_a --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.03 0.1 --parameter-range 0.2 0.3 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 单 GPU（完整生成当前数据集；可将 0 改为指定 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family swb_a --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.03 0.1 --parameter-range 0.2 0.3 --solver-backend cuda --gpus 0 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D --queue-log-root outputs/gpu_queue/generation

# run_e: depth ∈ [[0.55, 1.0], [1.3, 1.5]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family run_e --split interpolation --version 2 --size 500 --grid-shape 41x41x11 --parameter-range 0.55 1 --parameter-range 1.3 1.5 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 单 GPU（完整生成当前数据集；可将 0 改为指定 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family run_e --split interpolation --version 2 --size 500 --grid-shape 41x41x11 --parameter-range 0.55 1 --parameter-range 1.3 1.5 --solver-backend cuda --gpus 0 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D --queue-log-root outputs/gpu_queue/generation

# teb_b: minor_axis_out ∈ [[0.12, 0.18], [0.27, 0.32]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family teb_b --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.12 0.18 --parameter-range 0.27 0.32 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 单 GPU（完整生成当前数据集；可将 0 改为指定 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family teb_b --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.12 0.18 --parameter-range 0.27 0.32 --solver-backend cuda --gpus 0 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D --queue-log-root outputs/gpu_queue/generation

# bad_a: bend_angle_deg ∈ [[20.0, 45.0], [65.0, 90.0]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family bad_a --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 20 45 --parameter-range 65 90 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 单 GPU（完整生成当前数据集；可将 0 改为指定 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family bad_a --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 20 45 --parameter-range 65 90 --solver-backend cuda --gpus 0 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D --queue-log-root outputs/gpu_queue/generation

# lad_a: amplitude ∈ [[0.02, 0.08], [0.12, 0.18]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family lad_a --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.02 0.08 --parameter-range 0.12 0.18 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 单 GPU（完整生成当前数据集；可将 0 改为指定 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family lad_a --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.02 0.08 --parameter-range 0.12 0.18 --solver-backend cuda --gpus 0 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D --queue-log-root outputs/gpu_queue/generation

# srt_r: outlet_radius ∈ [[0.58, 0.68], [0.8, 0.9]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family srt_r --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.58 0.68 --parameter-range 0.8 0.9 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 单 GPU（完整生成当前数据集；可将 0 改为指定 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family srt_r --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.58 0.68 --parameter-range 0.8 0.9 --solver-backend cuda --gpus 0 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D --queue-log-root outputs/gpu_queue/generation

# sad_c: area_ratio_out ∈ [[0.6, 0.82], [1.08, 1.3]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family sad_c --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.6 0.82 --parameter-range 1.08 1.3 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 单 GPU（完整生成当前数据集；可将 0 改为指定 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family sad_c --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.6 0.82 --parameter-range 1.08 1.3 --solver-backend cuda --gpus 0 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D --queue-log-root outputs/gpu_queue/generation

# tbp_a: twist_angle_deg ∈ [[0.0, 30.0], [60.0, 90.0]]
# CPU（默认后端）
python src/data/generation/make_dataset_parametric_3d.py --family tbp_a --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0 30 --parameter-range 60 90 --solver-backend cpu --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# 单 GPU（完整生成当前数据集；可将 0 改为指定 GPU 编号）
python src/data/generation/make_dataset_parametric_3d.py --family tbp_a --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0 30 --parameter-range 60 90 --solver-backend cuda --gpus 0 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D --queue-log-root outputs/gpu_queue/generation

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
GPU 模式下每张卡一个独立进程，一次领取一个完整数据集；该数据集的端面构造、全部样本生成、合法性检测、保存及示例图完成后，才领取下一个数据集。
四张 GPU 最多同时生成四个数据集，每个数据集内部样本依次求解。`--workers` 仅控制 CPU 模式，不控制 GPU 并发数。只指定一个数据集时只使用第一张指定 GPU，其余 GPU 不参与该数据集的样本计算。

完整 CPU / GPU 命令已在上方“批量生成”“九类默认外推命令”“九类默认内插命令”逐组列出。

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--solver-backend` | `cpu` | `cpu` 保留原 SciPy 求解；`cuda` 使用 cuDSS GPU 稀疏直接求解。 |
| `--gpus` | CUDA 模式下为 `0` | 空格分隔的 GPU 编号；按 `CUDA_VISIBLE_DEVICES` 后的可见编号解释。必须搭配 `--solver-backend cuda`。 |
| `--workers` | `4` | 仅控制 CPU 模式的样本并行数。 |
| `--queue-log-root` | `outputs/gpu_queue/generation` | GPU 日志根目录；按 `3D/数据集名/generation.log` 分开保存。 |

GPU 使用 float64，保留端面9点、体网格19点的原 Winslow 离散格式和混合导数项。
上下端面各自求解，仍使用相同的 Coons＋三次样条位移参数面，侧面几何、数据格式和训练读取方式不变。
CPU/GPU 共用 NumPy/SciPy 的矩阵与右端项组装、TFI 初值、边界固定、外层迭代和停止条件；GPU 只将稀疏线性求解替换为 cuDSS 一般非对称直接分解。端面保留原 CPU 的边界消元，体网格保留原 CPU 的边界单位行。三个坐标共享一次分解，同时核验原始系统每个右端项的相对残差（内部上限 1e-9），失败时明确报错。原 GPU 的两个 `--linear-*` 迭代参数已删除。
CPU/GPU 的舍入误差可能不同，不承诺逐位相等；GPU 后端和求解诊断会保存到元数据。
GPU 不自动判断整张卡是否空闲，用户应指定允许使用的卡。Ctrl+C 会终止本次生成队列的全部子进程。
每个 GPU 工作进程仅写自己的数据集目录，不同数据集分别保存，不共享样本输出文件。
代表样本的边界面非法仍中断；代表样本的体网格非法在保存完整数据集后警告。
已有完整数据集仍按原规则跳过；切换后端不会自动覆盖它，需要重新生成时显式使用 `--force`。
cuDSS 已纳入总依赖：在项目根目录运行 `python -m pip install -r requirements.txt`（CUDA 12，固定 cuDSS 0.4 ABI），并使用兼容的 NVIDIA 驱动及 CUDA PyTorch。CPU 运行时不调用该库。未安装或加载失败时会明确报错，不静默回退。GPU 加速幅度取决于规模、稀疏分解、CPU 组装和传输成本，不保证小网格更快。

## GPU 日志与进度

CPU 进度继续显示在终端。GPU 数据集生成日志默认在 `outputs/gpu_queue/generation/3D/<数据集名称>/generation.log`，实时写入并刷新；可用 `--queue-log-root` 修改根目录。批量任务完成状态保存在日志根目录的 `queue_summary.json`。

GPU 日志只输出 `--timing-samples` 指定的前几个样本耗时（默认 3），总体完成数量每跨过 10% 输出一次；样本数不能整除 10 时输出实际进度。后续样本不逐个打印开始/完成或耗时。终端保留任务开始、完成、跳过和异常提示，不显示每个样本进度。异常堆栈保存到相应数据集日志。

```text
Sample 1/500: 12.345678 s
Sample 2/500: 12.456789 s
Sample 3/500: 12.567890 s
Progress: 50/500 (10%)
Progress: 100/500 (20%)
...
Progress: 500/500 (100%)
Dataset saved: .../data/raw/3D/swb_a_ext_2_500_21_21_11
```

以上耗时仅为格式示例。进度 100% 表示样本求解完成；任务须完成后续质量审计、写盘和示例图后才释放 GPU。完整数据集已存在则直接跳过，保留原生成日志；显式 `--force` 重新生成时覆盖该数据集旧日志。Ctrl+C 终止整个队列及当前数据集任务，不启动后续数据集。

### 三维生成的端面精度与检查规则

九类三维边界统一采用：两个端面分别求解 `201×201` 的二维 PDE，`cap_tolerance=1e-6`，`cap_max_iter=1000`；CPU 与 cuDSS GPU 均保留 float64。三维内部仍由三维 PDE 求解器求解，侧面几何、样条插值公式和硬边界约束不变。

每个参数区间选取两端及最接近中点的已有样本做完整合法性检查。内插的每个子区间分别选取；不额外插入样本。对这些代表样本检查端面原始 PDE 网格、同规模插值面、实际体网格规模的六面及最终体网格；删除原 `401×401` 加密插值检查。若显式修改端面规模，检查跟随 `cap_nx/cap_ny`，不再自动翻倍。非代表样本仅保留有限值、边界兼容性及求解收敛性检查。

代表边界面非法立即报错；最终体网格非法在完整保存数据集后警告。`metadata.json` 中的 `legality_check_sample_ids`、`quality_check_scope` 和 `cap_validation_policy` 记录检查范围；未检查不代表已证明合法。CPU 多进程与 GPU 整数据集队列采用相同规则，去除 CPU 重复预生成端面的预检。

已存在的完整数据集仍直接跳过，不自动覆盖或修改旧端面；需要重新生成时显式使用 `--force`。从已有数据集读取时继续使用已保存的参数面。上述改动不需要新增命令行参数。
