# 2D 九类数据集生成命令（内插与外推）

在项目根目录运行，以下18条命令显式给出样本数、节点规模、版本、参数范围及求解设置。

## 九类当前默认参数

样本数500，版本1，workers=4；二维为41×41，三维除run_e为41×41×11外均为21×21×11。表中的“几何函数默认值”用于单个边界示例；生成训练数据时会在指定区间均匀取样，并非只生成这个默认值。

| 边界公式 | 可变参数 | 几何函数默认值 | 外推训练范围 | 内插训练范围 | 节点规模 |
|---|---|---:|---|---|---|
| [air_t](boundaries/air_t.md) | `t` | 0.12 | `[[0.055, 0.6]]` | `[[0.055, 0.25], [0.35, 0.6]]` | 41×41 |
| [esa](boundaries/esa.md) | `inner_center_x` | 0.5 | `[[0.3, 0.7]]` | `[[0.3, 0.45], [0.55, 0.7]]` | 41×41 |
| [rse](boundaries/rse.md) | `semi_minor_axis` | 0.5 | `[[0.0, 0.4]]` | `[[0.0, 0.2], [0.3, 0.5]]` | 41×41 |
| [run](boundaries/run.md) | `depth` | 1.65 | `[[0.55, 1.5]]` | `[[0.55, 1.0], [1.3, 1.5]]` | 41×41 |
| [scc](boundaries/scc.md) | `depth` | 0.4 | `[[0.05, 0.4]]` | `[[0.05, 0.25], [0.35, 0.42]]` | 41×41 |
| [vsc](boundaries/vsc.md) | `constriction` | 0.65 | `[[0, 0.8]]` | `[[0, 0.3], [0.6, 0.85]]` | 41×41 |
| [tri_p](boundaries/tri_p.md) | `protrusion_distance` | 0.5 | `[[0.05, 0.4]]` | `[[0.05, 0.2], [0.3, 0.5]]` | 41×41 |
| [gsp_a](boundaries/gsp_a.md) | `amplitude` | 0.16 | `[[0.02, 0.18]]` | `[[0.02, 0.1], [0.15, 0.2]]` | 41×41 |
| [fca_a](boundaries/fca_a.md) | `arc_angle_deg` | 90 | `[[30.0, 120.0]]` | `[[30.0, 50.0], [70.0, 90.0]]` | 41×41 |

公式、固定几何参数和图片见表中的链接。以下命令是当前默认配置的显式快照；修改注册表后，批量命令读取新默认值，而显式命令仍使用写出的参数。

## 批量生成

```powershell
# 一次生成九类外推数据集
python scripts/data/generate_datasets.py --dimension 2 --families all --split ext --size 500 --version 1 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw

# 只生成九类内插数据集
python scripts/data/generate_datasets.py --dimension 2 --families all --split int --size 500 --version 1 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw

# 同时生成内插与外推（共18个数据集）
python scripts/data/generate_datasets.py --dimension 2 --families all --split both --size 500 --version 1 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-root data/raw
```

将`--split ext`改为`int`只生成九类内插集，改为`both`生成18个。`--dry-run`只列出任务。`--families`可直接写边界缩写列表；`new`只是最近新增三类的快捷筛选，与其余族共用同一注册表。

参数值在给定区间内均匀取样；无需随机种子。`--workers`是CPU并行求解进程数。已有同名完整数据集跳过。

## 九类默认外推命令（逐条运行）

```powershell
# air_t: t ∈ [[0.055, 0.6]]
python src/data/generation/make_dataset_parametric_2d.py --family air_t --split extrapolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.055 0.6 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-dir data/raw/2D

# esa: inner_center_x ∈ [[0.3, 0.7]]
python src/data/generation/make_dataset_parametric_2d.py --family esa --split extrapolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.3 0.7 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-dir data/raw/2D

# rse: semi_minor_axis ∈ [[0.0, 0.4]]
python src/data/generation/make_dataset_parametric_2d.py --family rse --split extrapolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0 0.4 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-dir data/raw/2D

# run: depth ∈ [[0.55, 1.5]]
python src/data/generation/make_dataset_parametric_2d.py --family run --split extrapolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.55 1.5 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-dir data/raw/2D

# scc: depth ∈ [[0.05, 0.4]]
python src/data/generation/make_dataset_parametric_2d.py --family scc --split extrapolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.05 0.4 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-dir data/raw/2D

# vsc: constriction ∈ [[0, 0.8]]
python src/data/generation/make_dataset_parametric_2d.py --family vsc --split extrapolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0 0.8 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-dir data/raw/2D

# tri_p: protrusion_distance ∈ [[0.05, 0.4]]
python src/data/generation/make_dataset_parametric_2d.py --family tri_p --split extrapolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.05 0.4 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-dir data/raw/2D

# gsp_a: amplitude ∈ [[0.02, 0.18]]
python src/data/generation/make_dataset_parametric_2d.py --family gsp_a --split extrapolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.02 0.18 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-dir data/raw/2D

# fca_a: arc_angle_deg ∈ [[30.0, 120.0]]
python src/data/generation/make_dataset_parametric_2d.py --family fca_a --split extrapolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 30 120 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-dir data/raw/2D

```

## 九类默认内插命令（逐条运行）

```powershell
# air_t: t ∈ [[0.055, 0.25], [0.35, 0.6]]
python src/data/generation/make_dataset_parametric_2d.py --family air_t --split interpolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.055 0.25 --parameter-range 0.35 0.6 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-dir data/raw/2D

# esa: inner_center_x ∈ [[0.3, 0.45], [0.55, 0.7]]
python src/data/generation/make_dataset_parametric_2d.py --family esa --split interpolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.3 0.45 --parameter-range 0.55 0.7 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-dir data/raw/2D

# rse: semi_minor_axis ∈ [[0.0, 0.2], [0.3, 0.5]]
python src/data/generation/make_dataset_parametric_2d.py --family rse --split interpolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0 0.2 --parameter-range 0.3 0.5 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-dir data/raw/2D

# run: depth ∈ [[0.55, 1.0], [1.3, 1.5]]
python src/data/generation/make_dataset_parametric_2d.py --family run --split interpolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.55 1 --parameter-range 1.3 1.5 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-dir data/raw/2D

# scc: depth ∈ [[0.05, 0.25], [0.35, 0.42]]
python src/data/generation/make_dataset_parametric_2d.py --family scc --split interpolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.05 0.25 --parameter-range 0.35 0.42 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-dir data/raw/2D

# vsc: constriction ∈ [[0, 0.3], [0.6, 0.85]]
python src/data/generation/make_dataset_parametric_2d.py --family vsc --split interpolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0 0.3 --parameter-range 0.6 0.85 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-dir data/raw/2D

# tri_p: protrusion_distance ∈ [[0.05, 0.2], [0.3, 0.5]]
python src/data/generation/make_dataset_parametric_2d.py --family tri_p --split interpolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.05 0.2 --parameter-range 0.3 0.5 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-dir data/raw/2D

# gsp_a: amplitude ∈ [[0.02, 0.1], [0.15, 0.2]]
python src/data/generation/make_dataset_parametric_2d.py --family gsp_a --split interpolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 0.02 0.1 --parameter-range 0.15 0.2 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-dir data/raw/2D

# fca_a: arc_angle_deg ∈ [[30.0, 50.0], [70.0, 90.0]]
python src/data/generation/make_dataset_parametric_2d.py --family fca_a --split interpolation --version 1 --size 500 --grid-shape 41x41 --parameter-range 30 50 --parameter-range 70 90 --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6 --timing-samples 3 --output-dir data/raw/2D

```

## 参数在哪里修改

`DATASET_FAMILIES`保存几何固定参数、内插/外推范围、默认网格规模；`SELECTED_DATASETS`决定无参数运行的任务、版本及显式网格规模。修改规模时同时检查这两个变量。`DEFAULT_GENERATION_OPTIONS`保存样本数、workers和体PDE求解设置。三维端面精度由每族`fixed_parameters`中的`cap_nx/cap_ny/cap_max_iter/cap_tolerance`控制，与体网格和网络输入规模分开。

## 常用参数速查

| 参数 | 含义 |
|---|---|
| `--family` | 单类生成入口使用边界缩写；批量入口使用 `--families`。 |
| `--split` | `ext/extrapolation`为外推训练集，`int/interpolation`为内插训练集；批量入口还支持`both`。 |
| `--size` | 几何样本数，默认500。 |
| `--version` | 数据集名称中的版本号，默认1。 |
| `--grid-shape` | 每个方向的节点数，不是单元数。 |
| `--parameter-range MIN MAX` | 可变几何参数的区间；重复填写可指定不连续区间。 |
| `--workers` | 独立样本的CPU并行求解进程数，默认4。 |
| `--winslow-max-iter` | 体/平面PDE迭代上限，默认1000。 |
| `--winslow-tolerance` | PDE求解停止容差，默认1e-6。 |
| `--timing-samples` | 输出详细生成时间的前几个样本，默认3。 |
| `--output-dir` / `--output-root` | 二维单类入口用前者；三维和批量入口用后者。 |
| `--dry-run` | 批量入口仅列出任务，不生成数据。 |

训练、评估及GPU选项见[正式实验命令](../../configs/2D/formal_external_commands.md)。

数据集生成后，预训练 ResMetaMesh 与评估无需扩展数据格式。九个默认外推数据集的四模型批量命令见 [2D 正式临时命令](../../configs/2D/formal_external_commands.md#一次性训练并评估全部九个数据集)。
