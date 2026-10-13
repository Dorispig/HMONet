# 评估体系、输出文件与新旧项目差异

更新日期：2026-09-16。本文描述已接入的新评估流程；`evaluation_comparison_air_t.md` 是修改前的检查记录。

## 目录

1. [运行方式](#运行方式)
2. [模型、边界误差与计时](#模型边界误差与计时)
3. [缓存和强制执行](#缓存和强制执行)
4. [最终输出结构](#最终输出结构)
5. [与旧项目的对比](#与旧项目的对比)
6. [验证范围](#验证范围)

## 运行方式

进入 MGNet 环境和项目目录：

```powershell
conda activate MGNet
cd E:\projects\HMONet
python scripts/evaluate/evaluate_2d.py --config configs/2D/evaluate.yaml
```

默认方法为 TFI、PDE、MeshONet、HMONet。二维默认 air_t 内插和外推：内插 t=0.275/0.30/0.325，外推 t=0.62/0.65；网格为 21×20、51×50、101×100。固定 m=0.02、p=0.4、c=1、外圆半径=1。

只比较 TFI 与 PDE：

```powershell
python scripts/evaluate/evaluate_2d.py --case air_t_int_0_1000_21_20 --values 0.30 --grid-shapes 21x20 --methods tfi,pde
```

该命令只生成这两种方法的结果和 `comparisons/01_tfi_pde/`，不会生成两个网络的占位文件。二维、三维评估图及局部图的横向顺序固定为 TFI、PDE、MeshONet、HMONet，与命令行或 YAML 中的填写顺序无关；只选择部分方法时保留这个相对顺序。三维汇总图以方法为列、切面为行。所有方法名称不区分输入大小写，输出使用统一标识。

覆盖网格、设备、内部投影、后处理和计时参数：

```powershell
python scripts/evaluate/evaluate_2d.py --case air_t_int_0_1000_21_20 --values 0.30 --grid-shapes 21x20,51x50 --methods tfi,pde,meshonet,hmonet --hmonet-mapping-mode current --hmonet-boundary-projection none --meshonet-boundary-projection inverse_distance --device cuda --inference-warmup-runs 3 --inference-timing-repeats 10
```

`--hmonet-mapping-mode both` 分别运行 current/coons_residual 两套匹配的模型，每套生成所选方法的比较组。也可以使用 `--hmonet-checkpoint`、`--meshonet-checkpoint` 指定路径；`--checkpoint` 仅适用于选中一个网络的情况。YAML 的 `hmonet_checkpoint` 可以是按 current/coons_residual 映射的路径字典。显式指定了错误模型或错误映射的 checkpoint 会报错，不会静默换模型。

缺失 checkpoint 时，默认明确报告缺失方法，再以其余所选方法生成比较。`--no-skip-missing-checkpoints` 则要求指定模型齐全；若没有任何方法可运行，会报错。一次比较默认列出四种方法，前提是两个模型的 checkpoint 都存在。

三维独立入口：

```powershell
python scripts/evaluate/evaluate_3d.py --config configs/3D/evaluate.yaml --case cah_t_int_0_1000_11_11_11 --values 0.20 --grid-shapes 11x11x11 --methods tfi,pde
```

用保存的几何样本检查已有数据：

```powershell
python scripts/evaluate/evaluate_2d.py --dataset data/raw/2D/air_t_int_0_1000_21_20 --parameter-index 0 --methods tfi,pde
```

YAML case 直接构造指定测试参数的精确边界，无需先生成测试数据集；`--dataset` 仅用于选择某条已保存几何。两者评估的 PDE 均由本次设置重新求解或从严格匹配的缓存读取，不把保存文件里的旧 TFI 当成 PDE 真值。

物理坐标局部图或 worst 质量局部图：

```powershell
python scripts/evaluate/evaluate_2d.py --case air_t_int_0_1000_21_20 --values 0.30 --grid-shapes 21x20 --metric intersecting_cell --local -0.55 0.55 -0.25 0.25
python scripts/evaluate/evaluate_2d.py --case air_t_int_0_1000_21_20 --values 0.30 --grid-shapes 21x20 --metric scaled_jacobian --center-on worst --cell-padding 2
```

独立局部图脚本读取现有结果，不重新执行网络：

```powershell
python scripts/evaluate/plot_result_zoom.py --parameter-dir "outputs/results/2D/air_t_int_0_1000_21_20/21x20/0.3" --plot quality --metric intersecting_cell --xlim=-0.55,0.55 --ylim=-0.25,0.25
python scripts/evaluate/plot_result_zoom_3d.py --batch-root "outputs/results/3D/cah_t_int_0_1000_11_11_11/11x11x11/0.2" --mode slice --slice z_mid --metric intersecting_cell --center-on worst
python scripts/evaluate/plot_result_zoom_3d.py --batch-root "outputs/results/3D/cah_t_int_0_1000_11_11_11/11x11x11/0.2" --mode volume --metric minimum_scaled_jacobian --center-on worst
```

二维 `--parameter-dir`、三维 `--batch-root` 会查找此参数下已有的比较组。三维 `--slice-coordinates physical` 为默认值，整体切片与局部切片用同样的物理投影：x_mid 显示 YZ，y_mid 显示 XZ，z_mid 显示 XY；可以显式选择 projected 使用公共 PCA 坐标。这样 box 的竖直截面不会投影成一条线。局部图同一指标、同一输出位置会覆盖前一次局部图。

## 模型、边界误差与计时

- 模型边界输入按 checkpoint 中的训练网格均匀采样，查询点按评估网格生成。这两个尺寸独立，所以训练 21×20 的网络可以查询 51×50 网格。
- HMONet 显式接收参数边界类；MeshONet 前向只接收边界点和计算域查询点。评估中的精确边界仍用于两个模型共同的边界误差和后处理基准。
- 每个网络前向只产生一份 raw 结果；direct/IDW 后处理操作它的副本，不覆盖原始数组。raw 与 postprocessed 分别统计和保存。
- HMONet 在二维、三维默认使用 `none`：跳过外部边界后处理，原始网络网格直接参与比较，`boundary_projection_seconds=0`。MeshONet 默认仍为 `direct`。为保持文件格式兼容，`none/` 结果仍有 `stage=postprocessed` 的记录，其数值与 raw 完全一致，不表示实际执行了修正。内部硬约束始终属于 HMONet 前向，仍计入网络时间。
- 原始边界误差为网络原始输出在逻辑边界节点上的预测值，与同参数处精确参数曲线/曲面的坐标之间的欧氏距离。保存最大、平均、RMS、分位数、最大误差位置及逐点数据。这里的最大值是所选边界节点上的最大误差，不是整条连续曲线的 Hausdorff 距离。
- `boundary_error_summary.json/csv` 同时包含 `stage=raw` 和 `stage=postprocessed` 两组记录。direct 后处理误差接近零并不能替代 raw 误差。
- 预热不计时；网络默认预热 3 次、测量 10 次，删除一次最大值和一次最小值，平均剩余 8 次。并列极值也只各删除一个样本；用户设为 1～2 次时使用普通平均。`--inference-warmup-runs` 和 `--inference-timing-repeats` 均可覆盖。输入采样、张量准备、输出搬回 CPU、绘图和文件保存不计入网络前向耗时；CUDA 使用事件计时。
- 强制生成或修改计时设置会触发重新测量。JSON 中 `inference_samples_seconds` 保留全部测量值，`inference_timing_statistic`、`inference_discarded_samples`、`inference_used_samples` 记录统计方式和样本数；CSV 也注明统计方式。前向内部的曲线参数打包、上传和求值属于网络调用成本；CUDA 事件跨度也可能包含 CPU 提交算子期间的设备空闲，不能直接解释为各 CUDA 计算核耗时之和。
- HMONet 每次前向计算投影，不缓存投影坐标。二维四个投影点为 left/right/bottom/top，三维六个投影点为 x_min/x_max/y_min/y_max/z_min/z_max。精确曲线/曲面求值与插值回退继续支持，训练和已有 checkpoint 兼容。权重计算合并重复算子；三维边界赋值仍保留原先后面覆盖前面的优先级。
- `current` 的权重计算与加权混合、`coons_residual` 的 TFI 混合与残差公式保持原定义。三维 Coons 前向只求一次六面投影，去掉此前未使用的重复求值。
- `hmonet_geometry_mode`/`--hmonet-geometry-mode` 支持 auto、direct、prepared、structured，默认 auto：单条常规评估使用 direct；批量或 CUDA Graph 使用 structured。prepared 在评估开始时打包几何参数；structured 额外按规则网格唯一的边/面参数求值，完整网格使用广播，分块网格使用局部索引。乱序或非笛卡尔查询不去重；不支持的自定义类回退普通求值。准备对象仅供评估，训练继续接收原参数边界类。
- `inference_batch_size`/`--inference-batch-size` 默认 1。兼容模型、尺寸和推理设置的参数案例可批量生成，仍逐个参数保存文件及边界误差。有效生成缓存不重复预测；批量失败回退逐案例处理。`inference_cuda_graph`/`--inference-cuda-graph` 默认 false，两种网络均可用；同一评估会话复用匹配的执行图，只更新边界输入、查询和几何系数，所有投影求值仍在图中执行。不同模型、网格坐标、批次形状或几何结构会重新捕获；最多保留 4 个图，不写 checkpoint/磁盘。未支持的后端/几何记录原因并回退。
- `curve_kernel_backend`/`--curve-kernel-backend` 支持 auto（默认）和 torch。auto 为支持的 CUDA float32/float64 曲线融合数学运算，兼容两段 NACA 的分段选择也在同一内核完成。二维和三维曲面所用曲线共用此路径；没有曲线拟合或投影值缓存。需要坐标梯度时自动使用原 Torch 公式，因此训练的高阶导数仍可计算。融合计算只有浮点舍入差异；已有 checkpoint 可直接复测。CUDA 编译库不可用时回退并记录原因，CPU/自定义曲线保留原路径。后端参与数值缓存、批量分组与 CUDA Graph 复用身份。
- 首次 CUDA 曲线内核编译/加载的时间另记 `curve_kernel_initialization_seconds`，可能需要数秒；普通预热期间的部分另记 `curve_kernel_warmup_initialization_seconds` 并加入准备时间，图捕获内的部分已经包含在 graph setup 中，不重复相加。auto 的 HMONet CUDA 推理至少预热一次，实际次数另记 `effective_inference_warmup_runs`。同进程后续调用复用已编译代码，不保存曲线值；`curve_kernel_calls` 和 `curve_kernel_fallbacks` 提供诊断。Windows/PyTorch 2.5 下，文件关闭后仅发布当前进程生成的曲线内核缓存，解决库内部在关闭文件前重命名失败导致重启后反复编译的问题；不替换已存在的缓存、不处理其他进程临时文件，默认使用 `%TEMP%/torch/kernels`。用户设置的 `PYTORCH_KERNEL_CACHE_PATH` 和禁用缓存选项仍受尊重。
- `network_inference_seconds` 为完整前向的上述截尾平均；批量时除以实际批次数，另记 `batch_inference_seconds`、`inference_batch_size` 和 `inference_time_basis`，不能把每条平均值作为单条延迟。`optimization_setup_seconds` 包括参数/索引准备、图捕获/更新及未重复计算的曲线内核初始化，`generation_with_setup_seconds` 为准备加前向，`total_model_seconds` 再加外部后处理。`geometry_setup_seconds`、`cuda_graph_setup_seconds`、`cuda_graph_used/reused` 和 `geometry_preparation` 提供明细。不计模型加载、通用输入采样、输出搬回 CPU 或绘图，只有最后一次测量输出回传。
- `paper_timing_table.csv` 中网络的 `Grid generation time (s)` 取完整前向时间，另列 Optimization setup time、Generation including setup、Inference batch size、Batch forward time 和 Timing basis；边界后处理单列。TFI/PDE 取各自生成耗时；生成缓存命中时保留原测量并标记复用，不把磁盘读取时间冒充推理时间。新增推理设置参与生成缓存签名，历史结果重新评估后采用新口径。
- 选择 PDE 时保存网络 raw/后处理结果相对本次 PDE 的逐点坐标 MSE 和逐单元最大顶点 MSE。未选择 PDE 时不暗中运行它，不生成这类参考误差。与旧评估相同，坐标分量取平均、不乘维度；当前训练固定网格 MSE×2 是训练侧另一种口径。
- PDE 异常不再退回 TFI。默认要求收敛，可用 `--no-require-pde-convergence` 保留真实但未收敛的迭代结果，摘要中明确记录状态。`hmonet_pde_iterations>0` 另生成 HMONet_PDE_refined，不覆盖原 HMONet。

## 缓存和强制执行

质量设置在两个维度的 `evaluate.yaml` 的 `evaluation` 中：

```yaml
compute_grid_quality: true
quality_metrics: both
quality_switches:
  intersecting_cell: false
  min_angle: true
```

`quality_metrics` 支持 `both`/`all`、`none` 或指标列表，布尔开关进一步筛选该列表。CLI 用 `--quality-metrics min_angle,scaled_jacobian` 选择，或 `--no-quality-intersecting-cell` 关闭单项，`--quality-intersecting-cell` 恢复该项开关。`--no-compute-grid-quality` 关闭所有质量计算，包括三维中间切片；保留各方法和比较组的网格图、网格文件，以及时间和边界误差等数值记录，不画误差颜色图。

二维及三维切片各支持 13 项指标，三维体网格支持 11 项，完整开关列表在各自 YAML 中。同名指标同时应用于三维体网格和切片；例如 `minimum_scaled_jacobian` 仅用于体网格，`scaled_jacobian` 仅用于四边形/切片。关闭指标不调用其专属计算，仅保留其他已选指标定义所必需的共享计算。部分指标的非法单元汇总只描述已检查范围。

质量选择纳入分析缓存签名，不改变网格生成参数。切换选择会清理受管理的旧质量文件并重做分析，已有匹配网格可以复用。使用 `--analysis-only` 可明确要求复用网格。关闭的指标不再用于评估内的质量局部图或 `worst` 搜索，此类任务明确跳过；坐标范围的网格局部图仍可生成。独立三维局部图脚本显式请求某项质量时只计算这一项；纯网格局部图不计算质量。

生成缓存按几何描述、查询网格、模型权重内容、结构参数、内部映射、PDE/计时设置和相关代码版本校验。不同方法分别缓存，改变所选方法组合可以复用已有方法的网格。原始网络缓存不绑定外部边界后处理，因此切换 direct/IDW 可以复用原始推理。

HMONet 的生成缓存还包含预计算开关，切换开关会重新执行 HMONet，避免拿另一种计时方式的缓存结果作比较；MeshONet 不受该开关影响。此处的磁盘缓存存放最终生成网格，不是投影点张量缓存。

分析缓存还记录图形开关和产物的尺寸/修改时间。丢失或修改质量文件时会修复分析产物，不需要重新前向。旧 `.complete` 不再参与判断。新评估器只清理自己记录在 manifest 中的文件，不递归删除整个参数目录或其他比较组。

| 开关 | 行为 |
| --- | --- |
| 无强制开关 | 自动复用匹配的生成和完整分析结果 |
| `--force-regenerate` | 重新生成；仅当生成的数组内容和分析配置均未变化时复用分析 |
| `--force-reanalyze` | 复用网格，重新计算质量并重画 |
| `--force` | 两个阶段均重新执行并覆盖对应组 |
| `--generation-only` | 只生成网格、原始边界误差和计时等生成记录 |
| `--analysis-only` | 必须已有匹配生成缓存；不允许偷偷执行模型或 PDE |
| `--no-plots` | 保存数值和表格，不生成图片 |
| `--save-distribution-plots` | 显式开启分布图；默认关闭 |

## 最终输出结构

以下是两个模型都存在、current 内部映射、HMONet 不做外部后处理、MeshONet 使用 direct、四方法按 TFI/PDE/MeshONet/HMONet 顺序时的完整结构。`<D>` 是 `2D` 或 `3D`，二维 `<grid>` 如 `21x20`，三维如 `11x11x11`。

```text
outputs/results/<D>/<dataset>/<grid>/<参数值>/
├── evaluation_parameters.json           # 几何参数、方法、checkpoint、有效配置
├── evaluation_timing.json               # 本次评估总时间与分方法时间
├── generation_timings.json              # 网络/后处理/TFI/PDE 时间明细
├── timings.csv
├── paper_timing_table.csv
├── boundary_error_summary.json          # raw + postprocessed 最大/平均等误差
├── boundary_error_summary.csv
├── results.json                         # 方法质量统计、边界误差、计时与来源
├── evaluation_summary.json
├── generated_meshes.npz                 # 本次所选方法处理后的网格
├── generated_raw_model_meshes.npz        # 两个网络未经后处理的预测
├── generation_metadata.json
├── .cache/
│   ├── TFI/{generation.npz,generation_metadata.json}
│   ├── PDE_Winslow/{generation.npz,generation_metadata.json}
│   ├── MeshONet/{generation.npz,generation_metadata.json}
│   └── HMONet/current/{generation.npz,generation_metadata.json}
├── TFI/                                # 公共方法文件见下方
├── PDE_Winslow/
├── MeshONet/
│   ├── raw/                            # 原始网格、质量及边界误差
│   └── direct/                         # 后处理后网格、质量及边界误差
├── HMONet/
│   └── current/
│       ├── raw/
│       └── none/                        # 与原始输出数值相同，不做外部修正
└── comparisons/
    └── current/hmonet_none__meshonet_direct/
        └── 01_tfi_pde_meshonet_hmonet/   # 公共比较文件见下方
```

HMONet coons_residual 改用对应同名层级；IDW 的单方法目录为 `inverse_distance/`，比较目录用简写 `idw`。`none/` 表示选择不后处理作为比较结果，`raw/` 总是保留独立原始结果。只有 TFI/PDE 时直接输出 `comparisons/01_tfi_pde/`；没有 HMONet 但包含 MeshONet 时输出 `comparisons/meshonet_<后处理>/01_<所选方法>/`。

每个公共方法目录（TFI、PDE_Winslow、MeshONet/raw、MeshONet/direct、HMONet/current/raw 等）包含：

```text
mesh.npy
mesh_data.npz                           # 坐标、单元连接及网格尺寸
mesh.png
mesh_metadata.json
analysis_metadata.json                  # 分析签名与完整性清单
grid_quality/
  mesh_data.npz
  quality_data.npz
  quality_data.csv
  quality_summary.json
  quality_summary.csv
  quality_<metric>.png
  invalid_cells.json
  invalid_cells.csv
  invalid_cells_visualization.png       # 二维
  mesh.png                              # 二维质量目录中的网格图
  mesh_comparison.png                   # 三维质量目录中的完整网格视图
boundary_error_summary.json             # 网络 raw/后处理目录
boundary_error_data.csv                 # 网络 raw/后处理目录
boundary_error.png                      # 网络 raw/后处理目录
raw_output_metadata.json                # 仅 raw，标明无后处理
```

每个公共比较组包含：

```text
metadata.json
comparison_mesh_data.npz                # 恰好包含实际所选方法，不复制方法目录
mesh_comparison.png
analysis_metadata.json
generation_timings.json
timings.csv
paper_timing_table.csv
boundary_error_summary.json
boundary_error_summary.csv
grid_quality/
  comparison_quality_data.npz
  comparison_quality_data.csv
  quality_summary.json
  quality_summary.csv                   # 包含各类非法单元数量、比例
  quality_<metric>.png                  # 所选方法横向排列、共同色标
  paper_quality_table.csv               # 二维
  paper_volume_quality_table.csv        # 三维
  pairwise_quality_differences.json      # 二维，两两质量差异
  pairwise_quality_differences.csv
  quality_<metric>_surfaces_and_slices.png # 三维体指标的表面/内部层概览
point_loss_data.npz                     # 选择 PDE 且有网络时
point_loss_data.csv
cell_loss_data.npz
cell_loss_data.csv
point_loss_summary.json
08_point_loss_values.png                # 逐点 MSE 与单元最大顶点 MSE，对数色标
09_point_loss_comparison.png            # 仅显式开启分布图时
zoom_views/                             # 配置或命令行要求局部图时
```

所有三维方法目录和三维比较组额外包含：

```text
surface_quality_summary.json
surface_quality_table.csv
self_intersection_table.csv            # 实际检测结果及比例，不再写死 0
boundary_faces/
├── x_mid/
├── y_mid/
└── z_mid/
    ├── metadata.json                   # 固定物理投影与指标定义
    ├── mesh.png
    ├── quality_<metric>.png
    ├── comparison_mesh_data.npz
    ├── comparison_quality_data.npz
    ├── quality_data.csv
    ├── quality_comparison.json
    ├── quality_summary.json
    ├── quality_summary.csv
    └── paper_surface_quality_table.csv
```

三个中间截面的目录结构相同。单方法截面还保存 `mesh.npy` 和 `quality_data.npz`。偶数层数在中间两层插值到逻辑参数 0.5，并非随意选取偏离中心的一层。

二维局部图输出原项目的 `*_zoom.png`、`*_overview_and_zoom.png` 和 `*_zoom_metadata.json`。三维按 `zoom_views/slices/<x_mid|y_mid|z_mid>/<metric>/` 或 `zoom_views/volume/<metric>/` 保存局部/整体/合并图、`local_mesh_data.npz`、逐单元质量 CSV、质量统计和 `zoom_metadata.json`。

PNG/PDF、分布图和生成/分析阶段均按开关决定实际文件是否产生。参数根目录的汇总记录描述最后一次调用；每个比较组保留自己的方法、checkpoint、投影和时间元信息，其他已存在的组不会被删除。

## 与旧项目的对比

| 项目 | 新项目现在的行为 | 与旧项目关系 |
| --- | --- | --- |
| YAML/CLI、case/values/grid_shapes | 已接入，二维三维分开执行 | 恢复 |
| 测试参数 | 根据留出的实际参数生成边界，无需索引训练样本 | 恢复 |
| 模型输入与查询尺寸 | 使用 checkpoint 训练尺寸，独立查询新网格 | 恢复 |
| HMONet 输入 | 精确参数曲线/曲面类 | 保留新设计，旧项目使用离散边界 |
| 计算域边界查询点 | 原始预测直接查询 0/1 端点，以测量精确参数处的网络边界误差 | 旧二维先将查询点夹到域内；新模型输入定义下保留此差异 |
| TFI | 精确参数几何生成，可查询给定参数网格；修复三维面/边/角组合 | 保留新设计 |
| PDE | 真正迭代、有收敛记录，失败不伪装为 TFI | 恢复并强化失败报告 |
| checkpoint 选择 | 按模型/数据集/划分/内部映射定位最新权重；支持完整 pt/带侧车参数的 pth | 恢复，新目录规则保留 |
| 两个模型 | 可一起运行；原始预测只执行一次，再做后处理 | 恢复 |
| 方法组合 | 严格按实际选择的方法生成一个比较组 | 用户指定的新行为；不恢复旧固定全部组合 |
| 内部映射 | current/coons_residual/both | 恢复；both 分别输出两套所选方法对比，不自动追加旧版全部消融组合 |
| 自适应模型比较 | 不加入 | 按此前要求保留差异，新项目无自适应训练 |
| 后处理 | none/direct/IDW；CLI 可覆盖、可设置幂次；所有网络模型默认 none | 默认比较各模型原始输出，避免外部边界修正干扰模型能力判断 |
| raw 输出 | 两个模型分别保存原始网格、质量、逐点边界误差 | 恢复并在参数/比较 JSON、CSV 中追加 raw 误差 |
| 边界误差 | 精确参数点作为基准，raw 与后处理分开报告 | 比旧离散边界基准更直接；不比较后处理误差来冒充网络精度 |
| 网络计时 | 预热、重复取中位数，网络与后处理分开 | 恢复 |
| 投影预计算 | 评估可预计算四个/六个投影点，两种内部映射均可读取；计时分列 | 新增评估优化；训练和模型权重保持原有方式 |
| 点/单元 PDE MSE | 仅在选择 PDE 且选择网络时输出；包含 raw、明细和对数色标图片 | 恢复并扩展 raw；方法改为横向排列，未选择 PDE 时不额外执行 |
| 网格质量 | 原二维13指标、原三维体指标和截面13指标；数据与图片保存 | 恢复 |
| 非法单元比例 | 各标志单独保存 count、cell_count、affected_cell_fraction | 对旧统计补充 |
| 自相交 | 二维自交/相交单元检测；三维截面内蕴对边相交及体单元翻转/退化统计 | 使用旧指标定义，不是完整三维任意两个体单元的碰撞证明 |
| 颜色、排列 | 方法横向、共享色标；整体与局部采用统一规则；二值指标0～1 | 恢复并统一 |
| 分布图 | 默认关闭，显式可开 | 用户指定差异，旧部分单方法图固定带分布 |
| 三维中间截面 | x_mid/y_mid/z_mid，偶数网格插值至0.5；整体/局部同坐标 | 恢复并固定视角；不会自动旋转每个方法 |
| 局部图 | 坐标矩形、worst、独立缓存读取、重复覆盖 | 恢复 |
| 缓存与强制 | 生成/分析独立、校验权重内容和输出完整性、可单独重做 | 恢复；不读取旧 `.complete` |
| 输出组织 | 参数值不带 t_ 前缀；模型按 raw/后处理区分；质量集中 grid_quality | 保留用户要求的新布局；不是旧路径逐字复制 |
| 摘要/CSV结构 | 保留核心文件名，补充 raw、比例和来源；质量统计统一逐指标行 | 新增字段和通用结构，第三方按旧列位置读取的脚本需要适配 |
| 几何工厂覆盖 | box、翼型孔圆柱及已实现二维参数边界 | 仍非旧项目全部几何：DSE、管道、stadium 精确工厂未实现，模板禁用并明确报错，绝不替换成 box |
| 三维数据真值 | 新生成 pde_mesh.npy 是实际 PDE，记录迭代信息 | 修复；旧错误数据不自动改写，需重新生成 |

本次没有改变训练循环、训练损失、网络结构或 checkpoint 保存体系。已有新项目二维 HMONet checkpoint 已实际用于评估；旧三维错误几何/TFI 数据上训练出的模型不能仅因代码修复就被认为具备正确真值训练，需要检查数据来源并酌情重新生成数据、训练。

## 验证范围

新增 `tests/test_evaluation_pipeline.py`，检查解析 box 的 TFI/PDE、非等尺寸索引、曲面边缘一致性、两维四方法身份、原始边界误差、训练/查询尺寸不同、完整/裸权重加载、checkpoint 按模型及映射筛选、权重变动使缓存失效、切换后处理复用原始推理、缺失文件修复、缺失模型/PDE异常不产生假结果、coons、局部图、分阶段运行及 PDE 后续平滑。

MGNet 环境下执行 `python -m unittest discover -s tests -v`，101 项测试全部通过（59.502 秒）。新增验证覆盖融合曲线的 float32/float64、分段连接点、几何参数更新、二维/三维模型参数梯度、曲线一阶/二阶导数回退、图重放查询更新及编译器不可用回退。现有测试覆盖训练、数据生成、评估、缓存及绘图。融合后的坐标允许浮点舍入误差，不能将历史优化的逐元素一致结论套用于本次融合计算。基准脚本为 `scripts/benchmark/benchmark_curve_kernels.py`，结果在 `outputs/benchmarks/curve_kernels_20260915/`；二维使用实际 air_t 检查点，三维使用随机权重，只用于性能检查。历史优化和撤回记录分别保留在 `outputs/benchmarks/evaluation_optimization_20260915/` 和 `outputs/benchmarks/projection_review_20260915/`。

人工试跑使用实际二维 air_t HMONet checkpoint、t=0.30、不同查询尺寸，包含 CPU 和 CUDA 路径；随后使用当前训练出的 MeshONet 与 HMONet checkpoint，完成 CUDA 上 t=0.30、7×6 查询网格的四方法评估及图片输出。三维 box 和翼型孔圆柱执行了实际 TFI/PDE，整体质量图、二维局部图、三维截面和体局部图已打开检查。当前配置中所有启用 case 的测试参数均完成边界构造及一致性检查。

实际二维四方法试跑中，MeshONet 原始边界最大/平均误差为约 0.00402836 / 0.00186489；HMONet 为约 9.97e-8 / 3.40e-8。两个模型的 direct 后处理误差均为 0，两套误差分开记录。这些数值只说明该模型、该参数、该查询网格的结果，不能代替全测试集的精度评估。

两维四方法接口还使用测试生成的小网络权重验证，其中三维尚未用生产训练权重做全量精度验证。没有遍历所有大网格与所有训练权重，也没有声称重新验证生产模型的泛化质量。测试、解析答案和实际试跑共同用于检查流程及文件真实性。
