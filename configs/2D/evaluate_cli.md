# 二维评估命令行使用说明

对应配置：[evaluate.yaml](evaluate.yaml)；训练说明：[train_cli.md](train_cli.md)。
入口：`scripts/evaluate/evaluate_2d.py`。本说明从原 YAML 顶部的命令行注释迁出，保留全部参数及示例。

运行前先执行：

```powershell
conda activate MGNet
cd E:\projects\HMONet
```

参数清单中的选项追加在 `python scripts/evaluate/evaluate_2d.py` 后使用；每一项都是独立选项，无需全部同时输入。完整命令示例可直接复制。

- 配置优先级：命令行 > `cases.<名称>.evaluation` > `evaluation` > 程序默认值。
- 布尔开关直接写名称，后面不加 `true/false`；`--no-` 前缀关闭。
- 多值使用英文逗号分隔、不加空格；路径包含空格时加双引号。
- 命令行覆盖仅对本次运行生效，不回写 YAML；网格大小表示点数。

## 目录

- [一、可直接复制的完整命令示例](#examples)
- [二、最常用：选择数据、参数、网格和评估方法](#cases)
- [三、常用：模型 checkpoint、内部映射和边界后处理](#models)
- [四、常用：选择计算哪些网格质量](#quality)
- [五、常用：图片输出、结果复用和强制重新运行](#outputs)
- [六、常用：二维物理坐标局部图和最差区域局部图](#local-views)
- [七、进阶：网络计时、显存分块和推理加速](#timing)
- [八、进阶：PDE 求解、收敛要求和 HMONet 后续平滑](#pde)
- [九、较少使用：模型配置回退、帮助和共享解析器中的三维专用选项](#other-options)

<a id="examples"></a>

## 一、可直接复制的完整命令示例

```powershell
python scripts/evaluate/evaluate_2d.py --config configs/2D/evaluate.yaml
```

含义：按 `evaluate.yaml` 中的 default_cases、各 case 的 values/grid_shapes 和 evaluation 配置评估。

```powershell
python scripts/evaluate/evaluate_2d.py --case air_t_int_0_1000_21_20 --values 0.3 --grid-shapes 51x50
```

含义：评估 air_t 内插模型对应的 t=0.3，输出 51×50 网格，方法和质量沿用 YAML。

```powershell
python scripts/evaluate/evaluate_2d.py --case air_t_int_0_1000_21_20 --values 0.275,0.3,0.325 --grid-shapes 21x20,51x50 --methods tfi,pde,meshonet,hmonet
```

含义：逐个参数值、逐个网格尺寸运行所选四种方法，并分别保存对比结果。

```powershell
python scripts/evaluate/evaluate_2d.py --case air_t_int_0_1000_21_20 --values 0.3 --grid-shapes 51x50 --no-compute-grid-quality
```

含义：不计算网格质量，保留网格图/网格数据、计时及边界误差等数值记录。

```powershell
python scripts/evaluate/evaluate_2d.py --case air_t_int_0_1000_21_20 --values 0.3 --grid-shapes 51x50 --quality-metrics min_angle,scaled_jacobian
```

含义：仅选择最小角和缩放 Jacobian；这两项的 quality_switches 还需为 true。

```powershell
python scripts/evaluate/evaluate_2d.py --case air_t_int_0_1000_21_20 --values 0.3 --grid-shapes 51x50 --quality-metrics both --no-quality-intersecting-cell
```

含义：选择所有已实现质量，再关闭单元间相交指标；其余指标遵守各自开关。

```powershell
python scripts/evaluate/evaluate_2d.py --case air_t_int_0_1000_21_20 --values 0.3 --grid-shapes 51x50 --analysis-only --quality-metrics scaled_jacobian --metric scaled_jacobian --local -0.55 0.55 -0.25 0.25
```

含义：要求复用完全匹配的已生成网格，重用/更新质量分析，并画指定物理坐标矩形的局部图。

```powershell
python scripts/evaluate/evaluate_2d.py --case air_t_int_0_1000_21_20 --values 0.3 --grid-shapes 51x50 --curve-kernel-backend torch --hmonet-geometry-mode direct --no-inference-cuda-graph --inference-warmup-runs 3 --inference-timing-repeats 10 --force-regenerate
```

含义：用普通 Torch 曲线求值重新生成并计时，避免融合内核初始化；预热 3 次、测量 10 次。

<a id="cases"></a>

## 二、最常用：选择数据、参数、网格和评估方法

- `--case air_t_int_0_1000_21_20`：选择 cases 下的一个案例名称；覆盖 evaluation.default_cases。
- `--case air_t_int_0_1000_21_20 --case air_t_ext_0_1000_21_20`：重复 --case 选择多个案例，不用逗号拼接案例名。
- `--values 0.275,0.3,0.325`：覆盖所选 case.values；修改该 case.parameter 对应的物理参数值，如 air_t 的 t。
- `--grid-shapes 21x20,51x50`：覆盖所选 case.grid_shapes；每项为 nx×ny，点数均至少为 2；PDE 要求至少为 3。
- `--methods tfi,pde,meshonet,hmonet`：覆盖 evaluation.methods，只运行/比较所选方法；pde_winslow 也是 pde 的别名。
- `--model both`：选择网络为 hmonet、meshonet 或 both；保留原方法列表里的 TFI/PDE；显式 --methods 优先。
- `--device cuda`：覆盖 evaluation.device；支持 auto、cpu、cuda 或 cuda:0 等设备标识。
- `--output-root outputs/results`：覆盖 evaluation.output_root；默认输出为 outputs/results/2D/<case>/<nx>x<ny>/<参数值>/。
- `--config configs/2D/evaluate.yaml`：指定评估 YAML 文件；省略时读取项目中的 `configs/2D/evaluate.yaml`。
- `--all-cases`：遍历所用配置中全部 enabled=true 的 case，替代 default_cases；禁用案例跳过，不要与 --case 同时使用。
- `--dataset data/raw/2D/air_t_int_0_1000_21_20`：改为读取已有数据集中的一个几何样本；也可填绝对路径或数据集名称。
- `--parameter-index 0`：指定 --dataset 模式下的样本索引，从 0 开始；不是物理参数 t 的数值。

数据来源说明：使用 --dataset 时不再使用 --case/--all-cases/--values，样本几何由索引确定。

数据集模式仍可用 --grid-shapes 修改查询网格；省略时使用数据集记录的网格大小。

分辨率说明：网络的边界输入点数来自 checkpoint 的训练网格，--grid-shapes 修改的是生成网格分辨率。

方法顺序说明：图中始终按 TFI、PDE、MeshONet、HMONet 排列；仅选部分方法时保留这个相对顺序。

<a id="models"></a>

## 三、常用：模型 checkpoint、内部映射和边界后处理

- `--hmonet-checkpoint "<HMONet checkpoint文件路径>"`：指定 HMONet 权重，覆盖 evaluation.hmonet_checkpoint。
- `--meshonet-checkpoint "<MeshONet checkpoint文件路径>"`：指定 MeshONet 权重，覆盖 evaluation.meshonet_checkpoint。
- `--checkpoint "<checkpoint文件路径>"`：给唯一选中的网络指定权重；可同时选择 TFI/PDE，但不能同时选择两个网络。
- `--hmonet-mapping-mode current`：覆盖 evaluation.hmonet_mapping_mode；可选 current、coons_residual、coons、both。

映射说明：coons 是 coons_residual 的别名；both 分别使用匹配的 current/coons_residual checkpoint 评估并输出。

- `--hmonet-boundary-projection none`：覆盖 evaluation.hmonet_boundary_projection；可选 none、direct、inverse_distance、idw。
- `--meshonet-boundary-projection none`：覆盖 evaluation.meshonet_boundary_projection；可选 none、direct、inverse_distance、idw，默认 none。

后处理说明：none 不修正网络输出，direct 替换边界节点，inverse_distance 按反距离加权传播边界修正；idw 为其别名。

内外投影说明：hmonet-mapping-mode 控制网络内部约束；两个 boundary-projection 参数控制网络输出后的处理。

- `--inverse-distance-power 2.0`：覆盖 evaluation.inverse_distance_power；修改 IDW 距离权重指数，必须大于 0。
- `--skip-missing-checkpoints`：设置 evaluation.skip_missing_checkpoints=true；缺少匹配权重时报告并跳过该网络。
- `--no-skip-missing-checkpoints`：设置 evaluation.skip_missing_checkpoints=false；缺少匹配权重时报错。
- `--include-tfi`：设置 evaluation.include_tfi=true，不再过滤已有方法列表中的 TFI；不会主动向列表补入 TFI。
- `--no-include-tfi`：设置 evaluation.include_tfi=false；未显式使用 --methods 时，从方法列表中去除 TFI。

权重说明：不指定 checkpoint 时自动寻找匹配数据集、模型及内部映射的最新权重；best.pt/best_model.pth 均可读取。

路径说明：上面的尖括号是需要替换的路径占位符，不能原样执行。

<a id="quality"></a>

## 四、常用：选择计算哪些网格质量

- `--compute-grid-quality`：设置 evaluation.compute_grid_quality=true，允许按列表和各指标开关计算质量。
- `--no-compute-grid-quality`：设置 evaluation.compute_grid_quality=false，关闭全部质量计算，网格图仍由 save_plots 控制。
- `--quality-metrics both`：覆盖 evaluation.quality_metrics；both/all 选择全部 13 项，none 选择零项，也支持英文逗号列表。

选择规则：实际计算项 = quality_metrics 选择列表与 quality_switches 中为 true 的指标的交集；总开关 false 时全部关闭。

名称规则：--quality-metrics/--metric 后面的指标名称使用下划线；各项命令行开关名称使用连字符。

- `--quality-intersecting-cell`：设置 evaluation.quality_switches.intersecting_cell=true，开启单元间相交/重叠指标。
- `--no-quality-intersecting-cell`：设置 evaluation.quality_switches.intersecting_cell=false，关闭单元间相交/重叠指标。
- `--quality-self-intersection`：设置 evaluation.quality_switches.self_intersection=true，开启单个四边形对边自相交指标。
- `--no-quality-self-intersection`：设置 evaluation.quality_switches.self_intersection=false，关闭单个四边形对边自相交指标。
- `--quality-concave-vertex-count`：设置 evaluation.quality_switches.concave_vertex_count=true，开启每个单元的凹顶点数量指标。
- `--no-quality-concave-vertex-count`：设置 evaluation.quality_switches.concave_vertex_count=false，关闭凹顶点数量指标。
- `--quality-flipping-cell`：设置 evaluation.quality_switches.flipping_cell=true，开启翻转单元指标。
- `--no-quality-flipping-cell`：设置 evaluation.quality_switches.flipping_cell=false，关闭翻转单元指标。
- `--quality-degenerate-cell`：设置 evaluation.quality_switches.degenerate_cell=true，开启退化单元指标。
- `--no-quality-degenerate-cell`：设置 evaluation.quality_switches.degenerate_cell=false，关闭退化单元指标。
- `--quality-min-angle`：设置 evaluation.quality_switches.min_angle=true，开启最小角指标，单位为度。
- `--no-quality-min-angle`：设置 evaluation.quality_switches.min_angle=false，关闭最小角指标。
- `--quality-max-angle`：设置 evaluation.quality_switches.max_angle=true，开启最大角指标，单位为度。
- `--no-quality-max-angle`：设置 evaluation.quality_switches.max_angle=false，关闭最大角指标。
- `--quality-scaled-jacobian`：设置 evaluation.quality_switches.scaled_jacobian=true，开启最小缩放 Jacobian 指标。
- `--no-quality-scaled-jacobian`：设置 evaluation.quality_switches.scaled_jacobian=false，关闭最小缩放 Jacobian 指标。
- `--quality-jacobian`：设置 evaluation.quality_switches.jacobian=true，开启单元中心 Jacobian 指标。
- `--no-quality-jacobian`：设置 evaluation.quality_switches.jacobian=false，关闭单元中心 Jacobian 指标。
- `--quality-skewness`：设置 evaluation.quality_switches.skewness=true，开启偏斜度指标。
- `--no-quality-skewness`：设置 evaluation.quality_switches.skewness=false，关闭偏斜度指标。
- `--quality-aspect-ratio`：设置 evaluation.quality_switches.aspect_ratio=true，开启边长长宽比指标。
- `--no-quality-aspect-ratio`：设置 evaluation.quality_switches.aspect_ratio=false，关闭边长长宽比指标。
- `--quality-orthogonality`：设置 evaluation.quality_switches.orthogonality=true，开启正交性指标。
- `--no-quality-orthogonality`：设置 evaluation.quality_switches.orthogonality=false，关闭正交性指标。
- `--quality-cell-area`：设置 evaluation.quality_switches.cell_area=true，开启单元面积指标。
- `--no-quality-cell-area`：设置 evaluation.quality_switches.cell_area=false，关闭单元面积指标。

计算说明：未选指标不单独计算/保存/绘图，但其他已选指标定义所必需的共享几何计算仍保留。

数量指标说明：凹单元、自相交、翻转等同时记录受影响单元比例；部分指标下的统计只代表已检查范围。

<a id="outputs"></a>

## 五、常用：图片输出、结果复用和强制重新运行

- `--save-plots`：设置 evaluation.save_plots=true，开启总绘图开关；具体格式仍由 save_png/save_pdf 控制。
- `--no-save-plots`：设置 evaluation.save_plots=false，不保存图片；所选质量的数值计算和文件输出仍执行。
- `--no-plots`：与 --no-save-plots 相同，是关闭总绘图开关的简写。
- `--save-png`：设置 evaluation.save_png=true，输出 PNG 图片。
- `--no-save-png`：设置 evaluation.save_png=false，不输出 PNG 图片。
- `--save-pdf`：设置 evaluation.save_pdf=true，输出 PDF 图片。
- `--no-save-pdf`：设置 evaluation.save_pdf=false，不输出 PDF 图片。
- `--save-distribution-plots`：设置 evaluation.save_distribution_plots=true，为已选质量输出分布图。
- `--no-save-distribution-plots`：设置 evaluation.save_distribution_plots=false，不输出分布图，仍可输出质量颜色图。
- `--force`：同时设置 evaluation.force_regenerate=true 和 force_reanalyze=true，重新生成并重做分析，覆盖对应旧结果。
- `--force-regenerate`：设置 evaluation.force_regenerate=true，重新运行所选方法生成网格并重新计时。
- `--force-reanalyze`：设置 evaluation.force_reanalyze=true，强制重做质量分析和图表；允许复用匹配的网格生成结果。
- `--analysis-only`：设置 evaluation.analysis_only=true，只使用严格匹配的网格生成缓存做分析；缓存不匹配或缺失时报错。
- `--generation-only`：设置 evaluation.generation_only=true，只生成/保存网格和生成记录，跳过质量分析、网格绘图及对比图。
- `--reuse-generated-results`：设置 evaluation.reuse_generated_results=true，复用匹配的已生成网格及其历史计时。
- `--no-reuse-generated-results`：设置 evaluation.reuse_generated_results=false，本次不复用已生成网格。
- `--reuse-analysis-results`：设置 evaluation.reuse_analysis_results=true，复用配置匹配且文件完整的质量分析和图表。
- `--no-reuse-analysis-results`：设置 evaluation.reuse_analysis_results=false，本次重新做质量分析和图表输出。

缓存说明：质量选择改变只使分析缓存失效；几何、checkpoint、网络计时等生成设置改变会使生成缓存失效。

冲突说明：analysis-only 不能与 generation-only、force-regenerate 或 force 同时启用。

分析模式说明：analysis-only 需要保留 reuse_generated_results=true；不复用网格时无法执行仅分析流程。

单向开关说明：force/force-regenerate/force-reanalyze/analysis-only/generation-only 传入时置为 true，省略时沿用 YAML。

<a id="local-views"></a>

## 六、常用：二维物理坐标局部图和最差区域局部图

- `--local -0.55 0.55 -0.25 0.25`：追加一个 evaluation.zoom_views 任务，设置物理坐标 X最小 X最大 Y最小 Y最大。
- `--xlim=-0.55,0.55`：设置新局部图任务的 xlim，修改横向物理坐标范围；应与 --ylim 一起使用。
- `--ylim=-0.25,0.25`：设置新局部图任务的 ylim，修改纵向物理坐标范围；应与 --xlim 一起使用。
- `--metric intersecting_cell`：设置新局部图任务的 metric；只影响局部图指标，不会自动开启该质量的计算开关。
- `--center-on worst`：设置新局部图任务的 center_on=worst，按 --metric 寻找最差单元附近区域。
- `--cell-padding 2`：设置新局部图任务的 cell_padding，修改最差单元/索引选择周围扩展的单元层数，默认 2。

局部图说明：--local 或 --xlim/--ylim 二选一；按坐标裁剪和 --center-on worst 也二选一。

负数说明：逗号参数以负数开头时使用等号形式，例如 --xlim=-0.55,0.55，避免被当成新的选项。

任务说明：只有提供坐标范围或 center-on 才会追加命令行局部图任务；单独 --metric 不会启动局部图绘制。

默认指标：二维命令行局部图默认使用 scaled_jacobian；已有 YAML zoom_views 任务会保留，新任务追加到列表末尾。

质量关闭说明：关闭的指标不用于质量局部图/worst 搜索，会输出跳过原因；指定坐标范围仍可输出网格局部图。

<a id="timing"></a>

## 七、进阶：网络计时、显存分块和推理加速

- `--inference-warmup-runs 3`：覆盖 evaluation.inference_warmup_runs，修改不计入前向统计的预热次数，允许 0，默认 3。
- `--inference-timing-repeats 10`：覆盖 evaluation.inference_timing_repeats，修改正式测量次数，至少 1，默认 10。

统计说明：测量至少 3 次时去掉一个最大值和一个最小值后取平均；测量 1～2 次时取普通平均，JSON 保留原始时间。


- `--query-batch-size 50000`：覆盖 evaluation.query_batch_size，限制每次前向的查询点数量，必须大于 0；不是几何样本批量。
- `--inference-batch-size 1`：覆盖 evaluation.inference_batch_size，修改兼容参数案例一起预测的样本数，必须大于 0。

批量说明：批量推理仍逐个物理参数保存文件；网络计时报告每个样本的平均值，不能当作单样本独立调用的延迟。

- `--curve-kernel-backend auto`：覆盖 evaluation.curve_kernel_backend；auto 融合支持的 CUDA 曲线公式，torch 使用普通 Torch 公式。

后端说明：仅有少量首次调用时可选 torch 避免融合初始化；反复调用可用 auto 摊薄初始化成本；几何公式保持一致。

Evaluation now passes the original boundary objects directly into the model.
There is no resident geometry preparation, Cartesian index preparation, CUDA Graph capture, or setup timing column.
`--hmonet-geometry-mode`, `--inference-cuda-graph` and `--no-inference-cuda-graph` have been removed.
Warmup runs follow `--inference-warmup-runs` exactly, including zero. With zero warmups, lazy runtime/kernel initialization can affect the first measured forward.
The generation column measures full network forward including internal projection; boundary postprocessing is recorded separately. It excludes model loading, quality analysis and plotting.

<a id="pde"></a>

## 八、进阶：PDE 求解、收敛要求和 HMONet 后续平滑

- `--winslow-max-iter 1000`：覆盖 evaluation.winslow_max_iter，修改 PDE_Winslow 最大迭代次数，必须大于 0。
- `--winslow-tolerance 1e-8`：覆盖 evaluation.winslow_tolerance，修改 PDE 收敛容差，必须大于 0。
- `--require-pde-convergence`：设置 evaluation.require_pde_convergence=true，PDE 达到迭代上限仍未收敛时报错。
- `--no-require-pde-convergence`：设置 evaluation.require_pde_convergence=false，允许保存未收敛的真实迭代结果并记录警告。
- `--hmonet-pde-iterations 0`：覆盖 evaluation.hmonet_pde_iterations；0 关闭，正整数表示以 HMONet 网格为初值的 PDE 最大迭代数。

平滑说明：只有已生成 HMONet 结果时才进行上述平滑，并额外输出 HMONet_PDE_refined，不覆盖原 HMONet 结果。

<a id="other-options"></a>

## 九、较少使用：模型配置回退、帮助和共享解析器中的三维专用选项

- `--train-config configs/2D/train.yaml`：指定重建网络时使用的回退训练配置文件；不启动训练，优先使用 checkpoint 自带配置。
- `-h`：显示二维入口支持的全部命令行参数并退出，不运行评估。
- `--help`：与 -h 相同；完整命令为 python scripts/evaluate/evaluate_2d.py --help。
- `--zlim=-1,1`：共享解析器接受此参数，但它只用于三维体网格局部图的 Z 坐标范围；二维评估不要设置。
- `--zoom-mode slice`：共享解析器可选 slice/volume，用于三维切片/体网格局部图；二维保留 slice，不要设 volume。
- `--slice z_mid`：共享解析器可选 x_mid/y_mid/z_mid，用于三维中间切片选择；二维没有切片，不需要设置。

三维专用说明：上面三项应在 scripts/evaluate/evaluate_3d.py 中配合三维案例使用，列在这里仅为完整说明共享入口参数。

CLI 范围说明：以上涵盖当前二维解析器的全部显式参数及正/反向开关；cases/fixed_parameters/zoom_views 等其余内容在 YAML 修改。


## PI-HMONet 与四模型选择

现支持 `attentionpihmonet`；`both` 保持 HMONet＋MeshONet，`all` 选择全部五种网络模型。
PI-HMONet 训练使用 `--pihmonet-mapping-mode current/coons_residual`；评估还可设为 `both`。
PI-HMONet 专属覆盖包括 pihmonet-pde-loss-weight、
pihmonet-interior-sample-count、pihmonet-fd-step、
pihmonet-jacobian-loss-weight 和 pihmonet-jacobian-minimum。
PI-MeshONet 同样支持 `--pimeshonet-*` 前缀的前四项；专属覆盖优先于原有 `--pi-*` 公共覆盖。
评估支持 `--pihmonet-checkpoint`、`--pihmonet-boundary-projection` 与 `--methods tfi,pde,pihmonet`。

Attention-PI-HMONet 使用 `--methods tfi,pde,attentionpihmonet`，可通过
`--attentionpihmonet-checkpoint` 指定检查点，或通过
`--attentionpihmonet-mapping-mode current/coons_residual` 选择内部映射。
训练评估串联时，模型和映射由 `--train-args` 指定，评估自动使用本次模型。
完整流程、公式及二维三维命令示例见项目中的 `scripts/train/pi_models.md`。
