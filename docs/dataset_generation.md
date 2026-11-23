# 参数化数据集与边界族指南

九类三维边界的两个端面分别采用201×201节点二维Winslow求解，再以精确Coons边界加三次样条位移定义连续参数面。四侧面在原参数域使用TFI，保留原解析几何；不做参数重分布、不复用上下端面求值。体内仍求解三维PDE。run_e默认41×41×11，其余八类默认21×21×11。所有新数据集名称版本为1，内部几何修订号为3。旧参数面数据与旧模型不能仅改名后复用。

本指南以**边界族**为单位组织数据，而非以每一个已生成的数据集目录组织。一个边界族可以对应多个内插/外推划分、样本规模、网格规模和版本；它们共享同一组几何公式与边界定向约定。这样既避免重复描述，又能让读者从边界定义直接复现实验。

## 注意事项与快速复现

1. 数据集名采用 `<short-name>_<int|ext>_<version>_<size>_<grid-shape>`：`int`/`ext` 分别表示内插/外推划分，`version` 表示几何定义或参数范围的人工版本，`size` 是几何样本数，`grid-shape` 是生成 PDE 网格的节点数。例如，`air_t_ext_1_500_41_41` 表示翼型孔边界族的外推版本 1、500 个几何样本和 `41×41` 节点网格。
2. 改变几何定义时应区分数据版本。本次按要求统一使用名称版本 `1`，通过元数据中的几何修订号 `3` 和参数面方案识别新格式；旧同名三维数据不会被当作新格式跳过，需显式 `--force` 重建。旧模型不能仅改名后用于新边界。
3. 二维和三维生成器均只对首、尾样本的 PDE 网格做合法性检查。若发现非法网格，先保存全部样本和质量报告（quality_audit.json），再发出警告，报告列出受检样本数、非法样本数和样本编号。PDE 求解器的收敛状态仍逐条记录。
   三维在求解前检查每个样本的六个规定边界面。若边界面有自交、翻转、退化或共享边不一致，立即报错。所有三维边界族（包括 run_e）均使用体 TFI 初始化后求解三维 Winslow 方程。若三维求解器发生数值异常并回退到体 TFI，该样本会标记为 tfi_fallback。
4. 同名输出目录不会被静默覆盖。需要重建时，先确认旧数据不再需要；三维仅在显式加入 `--force` 时替换已有目录。
5. 所有命令从项目根目录运行。数据默认写入 `data/raw/2D/` 或 `data/raw/3D/`，该目录不纳入 Git。生成后应保留元数据文件，保证 `dataset.npy` 与 `boundary_geometry.npy` 的样本顺序一致。

### 二维生成

新增三类二维和三类三维可一次批量生成：

```bash
python scripts/data/generate_datasets.py --dimension both --families new --split ext --size 500 --workers 4
```

```bash
python src/data/generation/make_dataset_parametric_2d.py \
  --family circle_airfoil_t --split extrapolation --version 1 \
  --size 500 --grid-shape 41x41 --parameter-range 0.055 0.60 \
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
  --size 500 --grid-shape 21x21x11 --parameter-range 0.04 0.30 \
  --workers 4 --winslow-max-iter 1000 --winslow-tolerance 1e-6
```

三维入口接受族名（或短名）、划分、版本、样本数、`NXxNYxNZ` 网格、PDE 求解选项和 `--parameter-range` 覆盖；`--workers N` 与二维一致，使用 `N` 个 CPU 进程并行求解互不依赖的样本，默认 `4`。覆盖范围时应同步递增 `--version`，以保证名称能够追溯几何定义。重复使用该选项可给出不连续的内插区间。完整训练—评估流程见根目录 [README](../README.md#从零复现九类外推数据集与正式实验)。

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
| `air_t` | 2-D | `circle_airfoil_t` | `t` | `[0.055,0.25] ∪ [0.35,0.60]` | `[0.055,0.60]` | `air_t_ext_1_500_41_41` |
| `esa` | 2-D | `eccentric_semiannulus` | `inner_center_x` | `[0.30,0.45] ∪ [0.55,0.70]` | `[0.30,0.70]` | `esa_ext_1_500_41_41` |
| `rse` | 2-D | `rect_top_semiellipse` | `semi_minor_axis` | `[0.00,0.20] ∪ [0.30,0.50]` | `[0.00,0.40]` | `rse_ext_1_500_41_41` |
| `run` | 2-D | `rounded_u_notch_compact` | `depth` | `[0.55,1.00] ∪ [1.30,1.50]` | `[0.55,1.50]` | `run_ext_1_500_41_41` |
| `scc` | 2-D | `smooth_constricted_channel` | `depth` | `[0.05,0.25] ∪ [0.35,0.42]` | `[0.05,0.40]` | `scc_ext_1_500_41_41` |
| `vsc` | 2-D | `variable_width_s_channel` | `constriction` | `[0.00,0.30] ∪ [0.60,0.85]` | `[0.00,0.80]` | `vsc_ext_1_500_41_41` |
| `tri_p` | 2-D | `symmetric_convex_tri` | `protrusion_distance` | `[0.05,0.20] ∪ [0.30,0.50]` | `[0.05,0.40]` | `tri_p_ext_1_500_41_41` |
| `gsp_a` | 2-D | `gear_spline` | `amplitude` | `[0.02,0.10] ∪ [0.15,0.20]` | `[0.02,0.18]` | `gsp_a_ext_1_500_41_41` |
| `fca_a` | 2-D | `four_convex_arcs` | `arc_angle_deg` | `[30,50] ∪ [70,90]` | `[30,120]` | `fca_a_ext_1_500_41_41` |
| `cah_t` | 3-D | `cylindrical_domain_with_airfoil_hole` | `thickness` | `[0.06,0.15] ∪ [0.25,0.35]` | `[0.08,0.34]` | `cah_t_ext_1_500_21_21_11` |
| `swb_a` | 3-D | `sidewall_deformed_block` | `sidewall_amplitude` | `[0.03,0.10] ∪ [0.20,0.30]` | `[0.04,0.30]` | `swb_a_ext_1_500_21_21_11` |
| `run_e` | 3-D | `rounded_u_notch_extrusion` | `depth` | `[0.55,1.00] ∪ [1.30,1.50]` | `[0.55,1.50]` | `run_e_ext_1_500_41_41_11` |
| `teb_b` | 3-D | `tapered_elliptic_bore_block` | `minor_axis_out` | `[0.12,0.18] ∪ [0.27,0.32]` | `[0.12,0.32]` | `teb_b_ext_1_500_21_21_11` |
| `bad_a` | 3-D | `bent_annular_duct` | `bend_angle_deg` | `[20,45] ∪ [65,90]` | `[25,85]` | `bad_a_ext_1_500_21_21_11` |
| `lad_a` | 3-D | `lobed_annular_duct` | `amplitude` | `[0.02,0.08] ∪ [0.12,0.18]` | `[0.04,0.16]` | `lad_a_ext_1_500_21_21_11` |
| `srt_r` | 3-D | `square_to_round_transition` | `outlet_radius` | `[0.58,0.68] ∪ [0.80,0.90]` | `[0.55,0.92]` | `srt_r_ext_1_500_21_21_11` |
| `sad_c` | 3-D | `variable_area_s_duct` | `area_ratio_out` | `[0.60,0.82] ∪ [1.08,1.30]` | `[0.55,1.35]` | `sad_c_ext_1_500_21_21_11` |
| `tbp_a` | 3-D | `twisted_blade_passage` | `twist_angle_deg` | `[0,30] ∪ [60,90]` | `[0,90]` | `tbp_a_ext_1_500_21_21_11` |

## 外推区间审计

外推区间不能只按整幅图中的位移大小判断。固定外包围盒、只改变孔洞或局部收缩的几何，其全局归一化位移可能很小，但局部尺度比已经发生数倍变化。以下结论同时检查了参数的物理变化幅度与默认离散规模下的端点网格合法性。

| 二维族 | 外推范围 | 更有意义的变化量 | 端点结论 |
| --- | --- | --- | --- |
| `air_t` | `0.055–0.60` | 翼型相对厚度约10.9倍 | 合理，合法 |
| `esa` | `0.30–0.70` | 内圆横移0.40，等于4个内圆半径 | 合理，合法 |
| `scc` | `0.05–0.40` | 最小通道间隙由0.90减至0.20 | 合理，合法 |
| `vsc` | `0–0.80` | 最小间隙由0.80减至0.16 | 合理，合法 |
| `run` | `0.55–1.50` | 凹槽深度约2.73倍 | 合理，合法 |
| `rse` | `0–0.40` | 顶部凸起由平直增至0.40 | 合理，合法 |
| `tri_p` | `0.05–0.40` | 尖点凸出距离8倍 | 合理，合法 |
| `gsp_a` | `0.02–0.18` | 齿形控制幅值9倍 | 合理，合法 |
| `fca_a` | 训练 `30°–120°`；外推评估 `125°、130°` | 大于90°时圆弧连接处形成凹角 | 120°的41×41节点PDE网格存在4个非法凹单元；生成结束后汇总合法性警告，125°、130°的质量需独立评估 |

| 三维族 | 外推范围 | 更有意义的变化量 | 端点结论 |
| --- | --- | --- | --- |
| `cah_t` | `0.08–0.34` | 翼型孔厚度约4.25倍 | 合理，体网格合法 |
| `swb_a` | `0.04–0.30` | 侧壁收缩幅值7.5倍 | 合理，体网格合法 |
| `run_e` | `0.55–1.50` | 与二维U槽相同，深度约2.73倍 | 使用 PDE 连续端面和三维 PDE；合法性取决于最终网格规模 |
| `teb_b` | `0.12–0.32` | 出口椭圆短轴约2.67倍 | 合理，体网格合法 |
| `bad_a` | `25°–85°` | 中心线弯曲角跨越60° | 变化充分，体网格合法 |
| `lad_a` | `0.04–0.16` | 波瓣幅值4倍 | 合理，体网格合法 |
| `srt_r` | `0.55–0.92` | 出口半径增加约67% | 合理，体网格合法 |
| `sad_c` | `0.55–1.35` | 出口/入口面积比约2.45倍 | 合理，体网格合法 |
| `tbp_a` | `0°–90°` | 总扭转角由0增至90° | 已由0°–38°放宽；90°端无翻转，最小体scaled Jacobian约0.250 |

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

### `symmetric_convex_tri`

上下边界为中心对称的三角折线，`protrusion_distance` 控制两个尖点的外凸距离，用于考察非光滑尖点附近的网格分布。

### `gear_spline`

四条边均为端点相容的周期 B 样条，`amplitude` 控制齿形幅值，用于引入多个连续高曲率区域。

### `four_convex_arcs`

四条边均为向外凸圆弧，`arc_angle_deg` 控制共同圆心角。候选范围 `30°–150°` 已做端点审计：150°的几何形变更明显，但 `41×41` Winslow 网格在四个尖角处出现4个翻转/凹单元，网格加密也不能消除。因此正式数据集保留经验证的 `30°–90°` 安全范围；候选图只说明几何差异，不作为合法训练目标。

![FCA候选30度与150度端点（仅几何展示）](figures/fca_a_candidate_30_150.png)

## 三维边界族

![三维边界族在外推区间两端的真实参数面示例](figures/boundary_families_3d.png)

![三维边界族在外推区间两端的逻辑中截面](figures/boundary_families_3d_sections.png)

图中每个视图展示由六个参数面组成的单块边界；蓝色/橙色仍对应外推区间的低端/高端。图可按下式重建：

```bash
python scripts/visualize/render_boundary_family_catalog.py --dimension 3 --output-dir docs/figures
```

### `cylinder_airfoil`

圆柱外域包围贯穿轴向的 NACA4 翼型孔。唯一变化参数为翼型相对厚度 `thickness`；边界由可直接计算的曲线挤出面与端面构成。

### `sidewall_deformed_block`

矩形块体的左右侧壁按正弦剖面对称向中心收缩，并沿 z 方向拉伸；唯一变化参数为无量纲幅值 `sidewall_amplitude`。实际公式为 `x=Lx*[1/2+(xi-1/2)*(1-a*sin(pi*eta))]`、`y=Ly*eta`、`z=z_min+Lz*zeta`，因此不是严格的椭圆柱面。`eta=0/1` 两面为矩形；`zeta=0/1` 两端为具有内凹侧边的平面区域。此处修正文档称谓，不更改现有几何或数据。

### `rounded_u_notch_extrusion`

二维 `run` 的四圆角深 U 槽截面沿轴向精确拉伸为三维块体；唯一变化参数为槽深 `depth`。每个固定 z 截面与二维 `rounded_u_notch` 的外轮廓完全一致，固定参数也相同：`width=3.0`、`height=2.0`、`notch_width=0.9`、`radius=0.25`。它不是 PI-MeshONet 的 Test case 4。

注意：`11×21×11` 和 `21×21×11` 的深凹网格并非均合法；端面合法、体 PDE 收敛、最终单元合法是三个不同条件。用 `python scripts/data/check_run_e_grids.py` 可复核训练两端 `0.55、1.50` 和评估两端 `1.52、1.55`，结果保存至 `reports/3D/run_e_pde_caps/grid_validation.json`。该检查只针对给定参数，不代表已生成并验证全部 500 个样本。

### `tapered_elliptic_bore_block`

矩形块体中包含轴向渐变的椭圆贯通孔；唯一变化参数为出口短半轴 `minor_axis_out`。外矩形采用四条标准直线段参数化，角点固定在逻辑周向参数的四等分位置，不再通过极坐标射线与矩形求交。该族提供非圆横截面连续变化的孔洞拓扑。

### `bent_annular_duct`

沿平面圆弧中心线扫掠的环形弯管，唯一变化参数为弯曲角 `bend_angle_deg`。它强调曲线中心线与截面环形结构的耦合。

### `lobed_annular_duct`

带轴向相位旋转的星形波瓣环形管，唯一变化参数为波瓣幅值 `amplitude`。该族引入周期性横截面变化，但不要求显式曲率或法向量作为网络输入。

### `square_to_round_transition`

矩形入口通过光滑轴向混合过渡为圆形出口，`outlet_radius` 控制出口半径。该族用于检验截面拓扑不变但形状类型显著变化的管道映射。

### `variable_area_s_duct`

截面沿平面 S 形中心线扫掠并连续改变面积，`area_ratio_out` 控制出口/入口面积比。该族同时包含中心线曲率和截面尺度变化。

### `twisted_blade_passage`

几何表示一对相邻叶片之间的单个流道。两个叶片面随展向改变安装角，`twist_angle_deg` 控制总扭转角；它不是实体叶片外包围盒。原 `0°–38°` 区间的几何变化偏弱，现扩大为 `0°–90°`。90°端的 `21×21×21` TFI 与 Winslow 网格均无翻转，最小体 scaled Jacobian 约为0.250。

![TBP扩大后的0度与90度端点](figures/tbp_a_endpoints_0_90.png)

### `localized_dimple_block`（仅几何生成器）

局部凹坑暂不登记为数据集族。解析映射为 `(Lx ξ, Ly η, ζ[H-d b(ξ)b(η)])`，其中 `b(s)=64s³(1-s)³` 且 `0≤d<H`。因此 Jacobian 行列式始终为正，映射与单位立方体双射。若只根据顶面的四条平直边做面 TFI，内部凹坑会被重建成平面，中心偏移量恰为 `d`；所以必须保存并直接求值完整解析凹坑面。

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

## 当前正式参数面与完整公式

两端面独立保存；旧曲面序列化只用于历史读取。每族公式和多参数可视化统一收录在[十八类边界](../outputs/boundaries/README.md)，二维与三维的最新范围及完整命令分别见[data/raw/2D/README.md](../data/raw/2D/README.md)和[data/raw/3D/README.md](../data/raw/3D/README.md)。

run_e默认规模按要求改为41×41×11。实测槽深0.55、1.50的端面及体网格通过；评估参数1.52、1.55的PDE体网格分别存在16、20个翻转单元，1.55的两个端面各有2个内凹四边形。数据集生成会拦截非法端面；评估会正常记录网格质量，不能把求解收敛或评估结束当作网格合法。61×21×11的通过记录保留在核验报告中。
