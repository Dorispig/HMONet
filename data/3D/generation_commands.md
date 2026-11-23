# 3D 九类数据集生成命令（内插与外推）

在项目根目录运行，以下18条命令显式给出样本数、节点规模、版本、参数范围及求解设置。

九类三维边界的两个端面分别采用201×201节点二维Winslow求解，再以精确Coons边界加三次样条位移定义连续参数面。四侧面在原参数域使用TFI，保留原解析几何；不做参数重分布、不复用上下端面求值。体内仍求解三维PDE。run_e默认41×41×11，其余八类默认21×21×11。三维新数据集名称版本为2，内部几何修订号为3。旧参数面数据与旧模型不能仅改名后复用。

## 九类当前默认参数

样本数500，版本2，workers=4；二维为41×41，三维除run_e为41×41×11外均为21×21×11。表中的“几何函数默认值”用于单个边界示例；生成训练数据时会在指定区间均匀取样，并非只生成这个默认值。

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

```powershell
# 一次生成九类外推数据集
python scripts/data/generate_datasets.py --dimension 3 --families all --split ext --size 500 --version 2 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw

# 只生成九类内插数据集
python scripts/data/generate_datasets.py --dimension 3 --families all --split int --size 500 --version 2 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw

# 同时生成内插与外推（共18个数据集）
python scripts/data/generate_datasets.py --dimension 3 --families all --split both --size 500 --version 2 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw
```

将`--split ext`改为`int`只生成九类内插集，改为`both`生成18个。`--dry-run`只列出任务。`--families`可直接写边界缩写列表；`new`只是最近新增三类的快捷筛选，与其余族共用同一注册表。

参数值在给定区间内均匀取样；无需随机种子。`--workers`是CPU并行求解进程数。已有同名完整数据集跳过。同名旧参数面数据会明确报错并保留原文件；确认替换时使用`--force`，否则使用新的输出目录。

## 九类默认外推命令（逐条运行）

```powershell
# cah_t: thickness ∈ [[0.08, 0.34]]
python src/data/generation/make_dataset_parametric_3d.py --family cah_t --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.08 0.34 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# swb_a: sidewall_amplitude ∈ [[0.04, 0.3]]
python src/data/generation/make_dataset_parametric_3d.py --family swb_a --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.04 0.3 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# run_e: depth ∈ [[0.55, 1.5]]
python src/data/generation/make_dataset_parametric_3d.py --family run_e --split extrapolation --version 2 --size 500 --grid-shape 41x41x11 --parameter-range 0.55 1.5 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# teb_b: minor_axis_out ∈ [[0.12, 0.32]]
python src/data/generation/make_dataset_parametric_3d.py --family teb_b --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.12 0.32 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# bad_a: bend_angle_deg ∈ [[25.0, 85.0]]
python src/data/generation/make_dataset_parametric_3d.py --family bad_a --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 25 85 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# lad_a: amplitude ∈ [[0.04, 0.16]]
python src/data/generation/make_dataset_parametric_3d.py --family lad_a --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.04 0.16 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# srt_r: outlet_radius ∈ [[0.55, 0.92]]
python src/data/generation/make_dataset_parametric_3d.py --family srt_r --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.55 0.92 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# sad_c: area_ratio_out ∈ [[0.55, 1.35]]
python src/data/generation/make_dataset_parametric_3d.py --family sad_c --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.55 1.35 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# tbp_a: twist_angle_deg ∈ [[0.0, 90.0]]
python src/data/generation/make_dataset_parametric_3d.py --family tbp_a --split extrapolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0 90 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

```

## 九类默认内插命令（逐条运行）

```powershell
# cah_t: thickness ∈ [[0.06, 0.15], [0.25, 0.35]]
python src/data/generation/make_dataset_parametric_3d.py --family cah_t --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.06 0.15 --parameter-range 0.25 0.35 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# swb_a: sidewall_amplitude ∈ [[0.03, 0.1], [0.2, 0.3]]
python src/data/generation/make_dataset_parametric_3d.py --family swb_a --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.03 0.1 --parameter-range 0.2 0.3 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# run_e: depth ∈ [[0.55, 1.0], [1.3, 1.5]]
python src/data/generation/make_dataset_parametric_3d.py --family run_e --split interpolation --version 2 --size 500 --grid-shape 41x41x11 --parameter-range 0.55 1 --parameter-range 1.3 1.5 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# teb_b: minor_axis_out ∈ [[0.12, 0.18], [0.27, 0.32]]
python src/data/generation/make_dataset_parametric_3d.py --family teb_b --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.12 0.18 --parameter-range 0.27 0.32 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# bad_a: bend_angle_deg ∈ [[20.0, 45.0], [65.0, 90.0]]
python src/data/generation/make_dataset_parametric_3d.py --family bad_a --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 20 45 --parameter-range 65 90 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# lad_a: amplitude ∈ [[0.02, 0.08], [0.12, 0.18]]
python src/data/generation/make_dataset_parametric_3d.py --family lad_a --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.02 0.08 --parameter-range 0.12 0.18 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# srt_r: outlet_radius ∈ [[0.58, 0.68], [0.8, 0.9]]
python src/data/generation/make_dataset_parametric_3d.py --family srt_r --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.58 0.68 --parameter-range 0.8 0.9 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# sad_c: area_ratio_out ∈ [[0.6, 0.82], [1.08, 1.3]]
python src/data/generation/make_dataset_parametric_3d.py --family sad_c --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0.6 0.82 --parameter-range 1.08 1.3 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

# tbp_a: twist_angle_deg ∈ [[0.0, 30.0], [60.0, 90.0]]
python src/data/generation/make_dataset_parametric_3d.py --family tbp_a --split interpolation --version 2 --size 500 --grid-shape 21x21x11 --parameter-range 0 30 --parameter-range 60 90 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw/3D

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
