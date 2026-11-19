# 论文多面板图片组装

`compose_paper_figure.py` 将已有的网格图或质量图按指定行列排版，输出可直接用于论文的 PNG 和 PDF。脚本提供两种互斥输入模式：

1. `--images`：直接拼接已有的 PNG、JPG、TIFF 或 WebP 图片。适用于二维、三维已有评估图的快速排版。
2. `--mesh-files`：读取二维 `mesh.npy` 后重新绘图。适用于需要统一坐标范围、精确边界、网格线样式，以及真正共享质量色标的正式论文图。

直接拼接模式不会修改原图中的边界、网格线、坐标轴或色标；如原质量图各自带有 colorbar，需先用 `--crop` 裁掉。二维重新绘图模式则由本脚本统一控制这些元素。

## 基本规则

- `--images` 和 `--mesh-files` 必须二选一。
- 图片或网格数量不得超过 `--rows × --cols`；少于格子数时，剩余格子自动隐藏。
- `--panel-titles`、`--panel-labels` 的数量必须等于面板数量；`--row-labels`、`--column-labels` 的数量必须分别等于行数、列数。
- `--mesh-files` 当前仅支持二维。每个网格应是评估输出的 `mesh.npy`，坐标形状为 `(NX×NY, 2)` 或 `(NY, NX, 2)`。
- 二维局部图使用 `--view-window XMIN XMAX YMIN YMAX` 指定物理坐标窗口；同一窗口将应用到全部面板，因此可直接比较局部单元。已有图片模式不能恢复物理坐标，应改用 `--crop` 进行像素裁剪。
- `--shape NX NY` 中的 `NX, NY` 是逻辑点数，例如 `101 101`，不是单元数。
- 输出路径没有扩展名时，脚本按 `--formats` 补充扩展名；有扩展名时，仅输出该指定文件。

## 参数说明

### 输入和布局

| 参数 | 类型/默认值 | 说明 |
| --- | --- | --- |
| `--images PATH ...` | 路径列表 | 直接拼接的已有栅格图片。与 `--mesh-files` 互斥。 |
| `--mesh-files PATH ...` | 路径列表 | 重新绘制的二维 `mesh.npy`。与 `--images` 互斥。 |
| `--rows` | 整数，必填 | 面板行数。 |
| `--cols` | 整数，必填 | 面板列数。 |
| `--mode` | `mesh` | `mesh` 绘制普通网格，`quality` 绘制单元质量场。直接图片模式中该参数只决定是否添加统一 colorbar。 |
| `--panel-titles TEXT ...` | 空 | 每个面板的小标题。 |
| `--panel-labels TEXT ...` | 空 | 每个面板左上角标签，例如 `"(a)" "(b)"`。 |
| `--row-labels TEXT ...` | 空 | 每一行左侧标签。 |
| `--column-labels TEXT ...` | 空 | 每一列上方标签。 |
| `--output PATH` | 必填 | 输出文件路径或文件名前缀。建议输出到 `paper/figures/`。 |
| `--formats png pdf` | `png` | 当 `--output` 无扩展名时生成的格式列表。 |

### 全局版式与文字

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--dpi` | `300` | PNG 输出分辨率。论文通常用 `300` 或 `600`。 |
| `--figure-width` | 自动 | 整张图宽度，单位英寸；不设时为 `cols × panel-width`。 |
| `--figure-height` | 自动 | 整张图高度，单位英寸；不设时为 `rows × panel-height`。 |
| `--panel-width` | `3.5` | 单面板默认宽度，单位英寸。 |
| `--panel-height` | `3.2` | 单面板默认高度，单位英寸。 |
| `--wspace` | `0.08` | 列间距，相对比例。 |
| `--hspace` | `0.12` | 行间距，相对比例。 |
| `--font-size` | `10` | 坐标轴刻度等普通文字字号。 |
| `--title-font-size` | `10` | 面板标题字号。 |
| `--label-font-size` | `11` | 面板编号、行列标签字号。 |
| `--background` | `white` | 图像背景颜色。 |
| `--transparent` | 关闭 | 输出透明背景；适合后续 LaTeX 排版。 |

### 已有图片模式：`--images`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--crop LEFT TOP RIGHT BOTTOM` | 不裁剪 | 对所有图片使用相同的归一化裁剪框，坐标范围均为 `[0, 1]`，原点在图片左上角。例如 `0 0 0.88 1` 裁掉右侧约 12%。 |
| `--show-image-axes` / `--no-show-image-axes` | 关闭 | 是否展示图片像素坐标轴。论文图片通常保持关闭。 |

当 `--images --mode quality` 且未指定 `--no-colorbar` 时，脚本会在整图右侧添加一个 colorbar。因为图片本身只是像素，必须用 `--vmin`、`--vmax` 明确其数值范围，才能使该色标具有真实物理含义。

### 二维网格重绘模式：`--mesh-files`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--shape NX NY` | 必填 | 网格逻辑点数。 |
| `--quality-data-files PATH ...` | 空 | 与网格一一对应的 `quality_data.npz`。仅质量图可选；不提供时将从网格重新计算指定指标。 |
| `--metric NAME` | 空 | 质量场名称；`--mode quality` 时必填，例如 `max_angle`、`scaled_jacobian`。 |
| `--dataset NAME` | 空 | 用于重建精确参数边界的数据集 case 名称，例如 `air_t_ext_1_500_21_21`。 |
| `--value FLOAT` | 空 | 数据集的单个可变几何参数，例如 `0.62`。当显示精确边界时必填。 |
| `--evaluate-config PATH` | `configs/2D/evaluate.yaml` | 包含 dataset case 定义的评估配置；通常无需修改。 |
| `--show-boundary` / `--no-show-boundary` | 默认显示 | 是否叠加精确参数边界。显示时必须同时提供 `--dataset` 与 `--value`。 |
| `--show-grid-lines` / `--no-show-grid-lines` | 默认显示 | 是否绘制网格单元边。关闭后，质量图为纯色块。 |
| `--show-axes` / `--no-show-axes` | 默认关闭 | 是否绘制物理坐标轴、刻度和 `x/y` 标签。 |
| `--shared-limits` / `--no-shared-limits` | 默认开启 | 所有面板是否使用同一物理坐标范围。论文方法对比图应保持开启。 |
| `--view-window XMIN XMAX YMIN YMAX` | 空 | 指定二维物理坐标局部窗口，覆盖 `--shared-limits`；全部面板使用该窗口。例如 `0.95 2.10 0.30 1.15`。仅可与 `--mesh-files` 使用。 |

### 网格、边界和质量场样式

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--mesh-linewidth` | `0.32` | 网格线宽；质量图中也用于单元边。 |
| `--mesh-color` | `#263238` | 普通网格线颜色。 |
| `--mesh-alpha` | `0.88` | 普通网格线透明度，范围 `0–1`。 |
| `--boundary-linewidth` | `1.6` | 精确边界线宽。 |
| `--boundary-color` | `#D62728` | 精确边界颜色；默认红色便于与网格区分。 |
| `--quality-edge-color` | `#303030` | 质量图单元边颜色。 |
| `--quality-alpha` | `1.0` | 质量场透明度，范围 `0–1`。 |
| `--cmap` | `viridis` | Matplotlib 色图名称，例如 `viridis`、`magma`、`turbo`。 |
| `--vmin` | 自动 | 质量图色标下限。方法比较应显式统一指定。 |
| `--vmax` | 自动 | 质量图色标上限。方法比较应显式统一指定。 |
| `--colorbar` / `--no-colorbar` | 质量图默认开启 | 是否添加 colorbar。整张质量图只生成一个色标。 |
| `--colorbar-label` | `Quality` 或指标名 | 色标标题，例如 `"Maximum angle (deg)"`。 |
| `--colorbar-width` | `0.022` | 统一色标相对于子图区域的宽度。 |
| `--colorbar-pad` | `0.025` | 色标和最右侧面板之间的距离。 |

## 命令行示例

以下示例中的路径请替换为实际评估结果路径。PowerShell 可使用反引号 `` ` `` 分行，也可将命令写成一行。

### 1. 拼接已有的二维或三维网格图

```powershell
python scripts/visualize/compose_paper_figure.py `
  --images outputs/results/2D/air_t_ext_1_500_21_21/101x101/0.62/PDE_Winslow/mesh.png `
           outputs/results/2D/air_t_ext_1_500_21_21/101x101/0.62/HMONet/mesh.png `
           outputs/results/2D/air_t_ext_1_500_21_21/101x101/0.62/MeshONet/mesh.png `
           outputs/results/2D/air_t_ext_1_500_21_21/101x101/0.62/PI-MeshONet/mesh.png `
  --rows 1 --cols 4 `
  --panel-titles PDE HMONet MeshONet PI-MeshONet `
  --panel-labels "(a)" "(b)" "(c)" "(d)" `
  --output paper/figures/air_t_mesh_062 `
  --formats png pdf --dpi 600
```

### 2. 将六张已有图片排为 2×3，并添加行列标签

```powershell
python scripts/visualize/compose_paper_figure.py `
  --images figure_1.png figure_2.png figure_3.png figure_4.png figure_5.png figure_6.png `
  --rows 2 --cols 3 `
  --column-labels "PDE" "HMONet" "PI-MeshONet" `
  --row-labels "Parameter = 0.62" "Parameter = 0.65" `
  --panel-labels "(a)" "(b)" "(c)" "(d)" "(e)" "(f)" `
  --panel-width 3.1 --panel-height 2.8 --wspace 0.04 --hspace 0.08 `
  --output paper/figures/air_t_two_parameters --formats png pdf
```

### 3. 直接拼接已有质量图，并裁掉每张原图自己的 colorbar

```powershell
python scripts/visualize/compose_paper_figure.py `
  --images pde_max_angle.png hmonet_max_angle.png meshonet_max_angle.png pimeshonet_max_angle.png `
  --rows 1 --cols 4 --mode quality `
  --crop 0 0 0.88 1 `
  --panel-titles PDE HMONet MeshONet PI-MeshONet `
  --cmap viridis --vmin 60 --vmax 180 `
  --colorbar-label "Maximum angle (deg)" `
  --output paper/figures/air_t_max_angle_raster --formats png pdf
```

此方式只适用于原图确实使用相同色图与同一数值范围的情形；否则应使用下面的网格重绘方式。

### 4. 从二维 `mesh.npy` 重新绘制普通网格图

```powershell
python scripts/visualize/compose_paper_figure.py `
  --mesh-files pde/mesh.npy hmonet/mesh.npy meshonet/mesh.npy pimeshonet/mesh.npy `
  --shape 101 101 --mode mesh `
  --dataset air_t_ext_1_500_21_21 --value 0.62 `
  --rows 1 --cols 4 `
  --panel-titles PDE HMONet MeshONet PI-MeshONet `
  --show-boundary --show-grid-lines --no-show-axes --shared-limits `
  --mesh-color "#1F2937" --mesh-linewidth 0.34 --mesh-alpha 0.90 `
  --boundary-color "#D62728" --boundary-linewidth 1.8 `
  --output paper/figures/air_t_mesh_rerendered --formats png pdf --dpi 600
```

### 5. 从二维网格生成共享色标的质量图

```powershell
python scripts/visualize/compose_paper_figure.py `
  --mesh-files pde/mesh.npy hmonet/mesh.npy meshonet/mesh.npy pimeshonet/mesh.npy `
  --quality-data-files pde/grid_quality/quality_data.npz `
                       hmonet/grid_quality/quality_data.npz `
                       meshonet/grid_quality/quality_data.npz `
                       pimeshonet/grid_quality/quality_data.npz `
  --shape 101 101 --mode quality --metric max_angle `
  --dataset air_t_ext_1_500_21_21 --value 0.62 `
  --rows 1 --cols 4 --panel-titles PDE HMONet MeshONet PI-MeshONet `
  --show-boundary --show-grid-lines --no-show-axes --shared-limits `
  --cmap viridis --vmin 60 --vmax 180 `
  --quality-edge-color "#303030" --mesh-linewidth 0.10 `
  --colorbar --colorbar-label "Maximum angle (deg)" `
  --output paper/figures/air_t_max_angle_shared --formats png pdf --dpi 600
```

省略 `--quality-data-files` 时，脚本会直接基于 `mesh.npy` 重算 `--metric`。当要复用评估期间的既有质量数值时，建议保留并提供该文件。

### 6. 生成无边界、无网格边的纯质量场

```powershell
python scripts/visualize/compose_paper_figure.py `
  --mesh-files hmonet/mesh.npy pimeshonet/mesh.npy `
  --shape 201 201 --mode quality --metric scaled_jacobian `
  --rows 1 --cols 2 --panel-titles HMONet PI-MeshONet `
  --no-show-boundary --no-show-grid-lines --no-show-axes `
  --cmap magma --vmin -1 --vmax 1 `
  --colorbar-label "Scaled Jacobian" `
  --output paper/figures/scaled_jacobian_clean --formats png
```

### 7. 从二维网格生成高曲率区域的局部图

局部图应使用 `--mesh-files`，以物理坐标保证所有方法展示同一位置。下例将四种方法限制到同一 `x∈[0.95, 2.10]`、`y∈[0.30, 1.15]` 的局部区域；精确边界仍会被绘制并自动裁切至窗口内。

```powershell
python scripts/visualize/compose_paper_figure.py `
  --mesh-files pde/mesh.npy hmonet/mesh.npy meshonet/mesh.npy pimeshonet/mesh.npy `
  --shape 101 101 --mode mesh `
  --dataset run_ext_1_500_41_41 --value 1.55 `
  --rows 1 --cols 4 --panel-titles PDE HMONet MeshONet PI-MeshONet `
  --view-window 0.95 2.10 0.30 1.15 `
  --show-boundary --show-grid-lines --show-axes `
  --mesh-linewidth 0.50 --boundary-linewidth 2.0 `
  --output paper/figures/run_notch_local_mesh --formats png pdf --dpi 600
```

质量局部图只需把上例中的 `--mode mesh` 改成 `--mode quality --metric max_angle`，并补充统一色标范围：

```powershell
python scripts/visualize/compose_paper_figure.py `
  --mesh-files pde/mesh.npy hmonet/mesh.npy meshonet/mesh.npy pimeshonet/mesh.npy `
  --shape 101 101 --mode quality --metric max_angle `
  --dataset run_ext_1_500_41_41 --value 1.55 `
  --rows 1 --cols 4 --panel-titles PDE HMONet MeshONet PI-MeshONet `
  --view-window 0.95 2.10 0.30 1.15 `
  --show-boundary --show-grid-lines --show-axes `
  --vmin 60 --vmax 180 --colorbar-label "Maximum angle (deg)" `
  --output paper/figures/run_notch_local_max_angle --formats png pdf --dpi 600
```

已有 PNG/JPG 局部图只能使用相同的像素裁剪，例如 `--crop 0.25 0.35 0.75 0.90`。它不保证不同图片对应同一物理区域，因此不建议将其作为正式的局部方法比较图。

### 8. 具有所有兼容可选参数的二维网格重绘示例

由于 `--images` 与 `--mesh-files` 互斥，且图片模式专有参数与网格重绘模式专有参数不能同时生效，下例覆盖了**网格重绘模式下全部可组合参数**：

```powershell
python scripts/visualize/compose_paper_figure.py `
  --mesh-files pde/mesh.npy hmonet/mesh.npy meshonet/mesh.npy pimeshonet/mesh.npy `
  --quality-data-files pde/grid_quality/quality_data.npz `
                       hmonet/grid_quality/quality_data.npz `
                       meshonet/grid_quality/quality_data.npz `
                       pimeshonet/grid_quality/quality_data.npz `
  --rows 2 --cols 2 --mode quality --shape 101 101 --metric max_angle `
  --dataset air_t_ext_1_500_21_21 --value 0.62 `
  --evaluate-config configs/2D/evaluate.yaml `
  --panel-titles PDE HMONet MeshONet PI-MeshONet `
  --panel-labels "(a)" "(b)" "(c)" "(d)" `
  --row-labels "Reference" "Operator models" `
  --column-labels "Classical" "Learned" `
  --output paper/figures/all_options_quality `
  --formats png pdf --dpi 600 `
  --figure-width 8.0 --figure-height 7.0 `
  --panel-width 3.6 --panel-height 3.0 --wspace 0.10 --hspace 0.15 `
  --font-size 9 --title-font-size 10 --label-font-size 11 `
  --background white --transparent `
  --show-boundary --show-grid-lines --show-axes --shared-limits `
  --mesh-linewidth 0.12 --mesh-color "#263238" --mesh-alpha 0.88 `
  --boundary-linewidth 1.8 --boundary-color "#D62728" `
  --quality-edge-color "#303030" --quality-alpha 1.0 `
  --cmap viridis --vmin 60 --vmax 180 `
  --colorbar --colorbar-label "Maximum angle (deg)" `
  --colorbar-width 0.022 --colorbar-pad 0.025
```

## 建议的论文作图流程

1. 用正式评估生成原始网格、数值汇总和需要的质量场数据；需要二次作图时，不要使用会删除 `mesh.npy` 的紧凑输出。
2. 普通方法比较图用 `--mesh-files --mode mesh --shared-limits --show-boundary` 重绘，确保所有方法坐标范围一致。
3. 质量比较图固定 `--cmap`、`--vmin` 与 `--vmax`，避免每张子图独立拉伸色标而掩盖差异。
4. 对跨参数、跨方法的大图，在同一命令里组装，统一 `--panel-width`、标题和 `(a)/(b)` 标签，避免后期手工拼图造成字体与线宽不一致。
5. 三维图片目前建议先用 `--images` 拼接；若后续需要三维切片/表面质量场的统一重绘与共享色标，应另行明确切片方向、体质量指标和渲染视角后扩展脚本。
