# 参数化数据集与边界族指南

本指南以**边界族**为单位组织数据，而非以每一个已生成的数据集目录组织。一个边界族可以对应多个内插/外推划分、样本规模、网格规模和版本；它们共享同一组几何公式与边界定向约定。这样既避免重复描述，又能让读者从边界定义直接复现实验。

## 注意事项与快速复现

1. 数据集名采用 `<short-name>_<int|ext>_<version>_<size>_<grid-shape>`：`int`/`ext` 分别表示内插/外推划分，`version` 表示几何定义或参数范围的人工版本，`size` 是几何样本数，`grid-shape` 是生成 PDE 网格的节点数。例如，`air_t_ext_1_500_21_21` 表示翼型孔边界族的外推版本 1、500 个几何样本和 `21×21` 节点网格。
2. 若改变几何公式、参数范围或边界方向，必须递增 `version`；不能用相同名称覆盖不同的几何定义。只改变样本数或网格规模时可保留版本号。
3. 二维生成器会在自定义参数范围的两个端点检查网格合法性。即使端点检查发出警告，数据仍会写入磁盘；此类数据仅用于诊断，不能直接作为论文正式数据。三维新族应先检查六个参数面的方向、共享边/角点兼容性与极端参数的 TFI 合法性。
4. 同名输出目录不会被静默覆盖。需要重建时，先确认旧数据不再需要；三维仅在显式加入 `--force` 时替换已有目录。
5. 所有命令从项目根目录运行。数据默认写入 `data/raw/2D/` 或 `data/raw/3D/`，该目录不纳入 Git。生成后应保留元数据文件，保证 `dataset.npy` 与 `boundary_geometry.npy` 的样本顺序一致。

### 二维生成

```bash
python src/data/generation/make_dataset_parametric_2d.py \
  --family circle_airfoil_t --split extrapolation --version 1 \
  --size 500 --grid-shape 21x21 --parameter-range 0.055 0.60 \
  --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6
```

常用选项：

- `--family`：下方“边界族目录”中的注册键；
- `--split`：`interpolation`/`int` 或 `extrapolation`/`ext`；
- `--version`、`--size`、`--grid-shape NXxNY`：控制数据集身份与规模；
- `--parameter-range MIN MAX`：覆盖可变参数范围；可重复给出多个区间；
- `--workers`：并行 PDE 真值生成进程数；`1` 表示串行；
- `--winslow-max-iter`、`--winslow-tolerance`：PDE 求解终止条件；
- `--output-dir`：显式指定输出根目录。

### 三维生成

```bash
python src/data/generation/make_dataset_parametric_3d.py \
  --family swb_a --split extrapolation --version 1 \
  --size 500 --grid-shape 11x21x11 --parameter-range 0.04 0.30 \
  --workers 4 --seed 20260905 --winslow-max-iter 1000 --winslow-tolerance 1e-6
```

三维入口接受族名（或短名）、划分、版本、样本数、`NXxNYxNZ` 网格、随机种子、PDE 求解选项和 `--parameter-range` 覆盖；`--workers N` 与二维一致，使用 `N` 个 CPU 进程并行求解互不依赖的样本，默认 `4`。覆盖范围时应同步递增 `--version`，以保证名称能够追溯几何定义。重复使用该选项可给出不连续的内插区间。完整训练—评估流程见根目录 [README](../README.md#从零复现六类外推数据集与正式实验)。

### 生成后检查

- Winslow PDE 是否按设定容差收敛；
- 端点参数下是否存在非正 Jacobian、翻转或自交；
- 曲线/曲面的参数方向、共享边和共享角点是否一致；
- 数据集名、元数据和实际参数范围是否一致；
- 后续训练是否明确记录该数据集名、随机种子、配置与 run tag。

## 边界族目录

目录条目是**几何类说明单元**，不是磁盘上的一个现存数据集。每个条目列出稳定的边界描述、唯一可变参数、复现所需注册键及一个命名示例；不会枚举当前机器已生成的全部数据集。

| 缩写 | 维度 | 生成器注册键 | 可变参数 | 推荐内插范围 | 推荐外推范围 | 数据集命名示例 |
| --- | ---: | --- | --- | --- | --- | --- |
| `air_t` | 2-D | `circle_airfoil_t` | `t` | `[0.055,0.25] ∪ [0.35,0.60]` | `[0.055,0.60]` | `air_t_ext_1_500_21_21` |
| `esa` | 2-D | `eccentric_semiannulus` | `inner_center_x` | `[0.30,0.45] ∪ [0.55,0.70]` | `[0.30,0.70]` | `esa_ext_1_500_21_21` |
| `rse` | 2-D | `rect_top_semiellipse` | `semi_minor_axis` | `[0.00,0.20] ∪ [0.30,0.50]` | `[0.00,0.40]` | `rse_ext_1_500_21_21` |
| `run` | 2-D | `rounded_u_notch_compact` | `depth` | `[0.55,1.00] ∪ [1.30,1.50]` | `[0.55,1.50]` | `run_ext_1_500_41_41` |
| `scc` | 2-D | `smooth_constricted_channel` | `depth` | `[0.05,0.25] ∪ [0.35,0.42]` | `[0.05,0.40]` | `scc_ext_1_500_41_41` |
| `vsc` | 2-D | `variable_width_s_channel` | `constriction` | `[0.00,0.30] ∪ [0.60,0.85]` | `[0.00,0.80]` | `vsc_ext_1_500_41_41` |
| `cah_t` | 3-D | `cylindrical_domain_with_airfoil_hole` | `thickness` | `[0.06,0.15] ∪ [0.25,0.35]` | `[0.08,0.34]` | `cah_t_ext_1_500_11_21_11` |
| `swb_a` | 3-D | `sidewall_deformed_block` | `sidewall_amplitude` | `[0.03,0.10] ∪ [0.20,0.30]` | `[0.04,0.30]` | `swb_a_ext_1_500_11_21_11` |
| `ugc_d` | 3-D | `rounded_u_groove_extrusion` | `groove_depth` | `[0.30,0.65] ∪ [1.00,1.30]` | `[0.30,1.30]` | `ugc_d_ext_1_500_11_21_11` |
| `teb_b` | 3-D | `tapered_elliptic_bore_block` | `minor_axis_out` | `[0.12,0.18] ∪ [0.27,0.32]` | `[0.12,0.32]` | `teb_b_ext_1_500_11_21_11` |
| `bad_a` | 3-D | `bent_annular_duct` | `bend_angle_deg` | `[20,45] ∪ [65,90]` | `[25,85]` | `bad_a_ext_1_500_11_21_11` |
| `lad_a` | 3-D | `lobed_annular_duct` | `amplitude` | `[0.02,0.08] ∪ [0.12,0.18]` | `[0.04,0.16]` | `lad_a_ext_1_500_11_21_11` |

## 二维边界族

![二维边界族在外推区间两端的真实参数边界示例](figures/boundary_families_2d.png)

图中蓝色/橙色分别由各注册族外推区间的低端/高端参数直接绘制。可用下述命令重新生成，不依赖任何已有数据集：

```bash
python scripts/visualize/render_boundary_family_catalog.py --dimension 2 --output-dir docs/figures
```

### `circle_airfoil`

外圆域中包含 NACA4 翼型孔，并由两条连接线完成单块参数边界。唯一变化参数为相对厚度 `t`；外圆半径、弯度、弯度位置和弦长保持固定。该族用于检验内部封闭曲线附近的边界贴合与间隙分辨能力。

### `eccentric_semiannulus`

内外上半圆与两条水平连接线形成偏心半环。唯一变化参数为内圆圆心横坐标 `inner_center_x`，内、外半径保持固定。该族强调局部窄隙和曲率变化的耦合。

### `top_semiellipse`

矩形的上边界由半椭圆替代，唯一变化参数为椭圆短半轴 `semi_minor_axis`。它提供从近似平顶到强弯曲顶边的连续变化。

### `rounded_u_notch`

矩形上边界包含带精确圆角的深 U 形凹槽；唯一变化参数为凹槽深度 `depth`。宽度、高度、凹槽宽度和圆角半径固定。该族用于测试高曲率凹陷与局部正交性的保持。

### `constricted_channel`

上下光滑相向凹陷形成通道，唯一变化参数为凹陷深度 `depth`，从而控制最小间隙。该族适合评估窄通道内的网格压缩和最差单元质量。

### `s_channel`

由 Bézier 曲线构成的变宽 S 形通道；唯一变化参数为收缩量 `constriction`，弯曲控制量固定。其边界具有整体弯曲和局部通道宽度同时变化的特征。

## 三维边界族

![三维边界族在外推区间两端的真实参数面示例](figures/boundary_families_3d.png)

图中每个视图展示由六个参数面组成的单块边界；蓝色/橙色仍对应外推区间的低端/高端。图可按下式重建：

```bash
python scripts/visualize/render_boundary_family_catalog.py --dimension 3 --output-dir docs/figures
```

### `cylinder_airfoil`

圆柱外域包围贯穿轴向的 NACA4 翼型孔。唯一变化参数为翼型相对厚度 `thickness`；边界由可直接计算的曲线挤出面与端面构成。

### `sidewall_deformed_block`

矩形块体的左右侧壁对称地向中心收缩，并沿前后方向贯通为凹椭圆柱面；唯一变化参数为无量纲收缩幅值 `sidewall_amplitude`。前后端面保持平面矩形，左右侧壁在每个前后截面均内凹；该族用于检验对称窄通道下的三维映射稳定性。

### `rounded_u_groove_extrusion`

二维圆角 U 形凹槽沿轴向延伸为三维凹槽块体；唯一变化参数为槽深 `groove_depth`。它不是 PI-MeshONet 的 Test case 4，而是与二维 U 形凹槽同源的轴向延伸几何。

### `tapered_elliptic_bore_block`

矩形块体中包含轴向渐变的椭圆贯通孔；唯一变化参数为出口短半轴 `minor_axis_out`。该族提供非圆、非同轴、局部横截面连续变化的孔洞拓扑。

### `bent_annular_duct`

沿平面圆弧中心线扫掠的环形弯管，唯一变化参数为弯曲角 `bend_angle_deg`。它强调曲线中心线与截面环形结构的耦合。

### `lobed_annular_duct`

带轴向相位旋转的星形波瓣环形管，唯一变化参数为波瓣幅值 `amplitude`。该族引入周期性横截面变化，但不要求显式曲率或法向量作为网络输入。

## 参数边界接口与可复现性

二维 `BoundaryCurveSet2D` 保存四条逻辑侧，三维 `BoundarySurfaceSet3D` 保存六个逻辑面。曲线/曲面由其原始参数直接求值；训练时的离散边界点仅作为编码器输入，不能替代精确边界实体。自定义族必须同时满足：

- 所有二维逻辑角点或三维共享边、共享角点兼容；
- 参数方向与逻辑边/面一致；
- NumPy 与 Torch 求值路径在相同参数上给出一致的几何；
- 若需要 GPU 可微投影，自定义实体实现 `evaluate_torch`；
- 新增族通过极端参数的 PDE、局部 Jacobian 和非局部相交检查。

```python
from src.data.geometry.parametric_curve_2d import BoundaryCurveSet2D

# boundary 由注册函数构造；xi、eta 可为批量 Torch 张量。
projections = boundary.project_torch(xi, eta)  # [B, N, 4, 2]
```

曲线的 `derivative` 返回相对于全局 `[0,1]` 参数的导数。驻留的几何对象应每个 batch 构造一次，并在多个前向/损失调用间复用；修改几何参数、设备或数据类型后需重建对象。
