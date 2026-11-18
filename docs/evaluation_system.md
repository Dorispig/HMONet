# 评估体系、指标与输出结构

本文只描述当前评估流程。历史修复记录和旧项目差异不作为当前行为依据；最终参数以入口 `--help` 和 `configs/2D|3D/evaluate.yaml` 为准。

## 运行方式

```bash
# 二维
python scripts/evaluate/evaluate_2d.py --case air_t_ext_1_500_41_41 --values 0.62 0.65 --grids 101 201 401 --methods tfi,pde,meshonet,pimeshonet,hmonet

# 三维
python scripts/evaluate/evaluate_3d.py --case swb_a_ext_1_500_11_21_11 --values 0.02 0.34 --grids 11x21x11 21x41x21 --methods tfi,pde,meshonet,pimeshonet,hmonet
```

三维显式网格按 `n_x×n_y×n_z` 写为一个参数，例如 `11x31x11`；只有纯数字 `21 31` 才被解释为两个立方网格。旧式 `--grid-shapes 11x31x11,21x61x21` 仍兼容。只选传统方法时不会生成模型占位文件。方法名称不区分输入大小写；图中顺序固定为 TFI、PDE、MeshONet、PI-MeshONet、HMONet、QG-HMONet。显式 checkpoint 必须与模型和 HMONet 映射模式匹配，错误身份会报错，不会静默替换。

YAML case 可直接构造指定参数的精确边界；`--dataset`/`--parameter-index` 用于选择已保存样本。PDE 结果由当前设置求解或从严格匹配缓存复用，不以旧 TFI 文件冒充 PDE。默认要求 PDE 收敛；允许保留未收敛结果时必须在摘要中明确状态。

## raw、边界后处理与边界误差

- 模型边界输入尺寸由 checkpoint 的训练配置确定，查询网格可以使用不同分辨率；
- 每次网络前向只生成一份 raw 网格，direct/IDW 后处理操作副本，不覆盖 raw；
- HMONet 的内部硬约束属于模型前向，不等同于外部边界后处理；
- raw 和 postprocessed 分别计算并保存边界误差；
- 边界误差是所选逻辑边界节点与精确参数边界对应点的欧氏距离，不是连续曲线/曲面的 Hausdorff 距离；
- 后处理误差接近零不能代替模型 raw 边界精度。

## 质量、相交与参考误差

`quality_metrics` 和各指标开关共同决定实际计算内容，CLI 可使用 `--quality-metrics` 或单项开关覆盖。`--no-compute-grid-quality` 关闭质量计算，但仍可保留网格、时间和边界误差。关闭的指标不会用于局部 worst 搜索。

质量数值计算与图片输出分离：默认仍完整保存所有启用指标的汇总，但只生成单方法网格图和方法并排网格图，不生成质量热图、分布图或质量局部图。需要图片时显式指定：

```bash
# 仅生成 scaled Jacobian 与最大角的单方法图、方法对比图
python scripts/evaluate/evaluate_2d.py ... --quality-plots scaled_jacobian max_angle

# 生成当前启用的全部质量图（含已有的质量对比图）
python scripts/evaluate/evaluate_2d.py ... --quality-plots all
```

消融实验入口使用相同参数，例如：`python scripts/experiments/run.py ex1 baseline all --quality-plots all`。

三维时按网格维度路由所选指标：公共指标（`max_angle`、`min_angle`、`skewness`、`orthogonality`、`degenerate_cell`）同时计算并绘制六面体体网格和物理中间切片；六面体专用指标仅用于体网格，四边形专用指标仅用于切片。因而，例如 `--quality-plots minimum_scaled_jacobian` 不会错误地产生切片指标图，而 `--quality-plots scaled_jacobian` 不会要求体网格计算该二维指标；`--quality-plots all` 会分别生成两类网格上全部启用的图。

二维评估报告角度、Jacobian/Scaled Jacobian、偏斜、长宽比、正交性及非法/翻转/退化/凹/相交单元。三维评估报告六面体体质量、非法状态，并在配置要求时统计中间截面质量。三维截面的相交检测不等价于任意两个六面体的完整碰撞证明，解释结果时应注明检测范围。

选择 PDE 且存在网络方法时，保存网络结果相对本次 PDE 网格的逐点坐标 MSE 和逐单元最大顶点 MSE。坐标分量按评估定义取平均；不得与采用不同维度缩放的训练损失直接混用。

## 计时口径

网络计时执行预热和重复测量；默认去掉一次最大值和一次最小值后求均值，测量次数不足时使用普通均值。模型加载、通用输入准备、输出回传、绘图和文件写入不计入纯前向时间；模型内部参数边界求值和硬约束属于前向成本。

报告同时区分：

- 网络前向时间；
- 外部边界后处理时间；
- 优化/内核初始化时间；
- 包含准备的网格生成时间；
- TFI/PDE 传统方法时间。

缓存命中时保留原始测量并标记复用，不能把磁盘读取时间当成推理时间。不同方法必须在同一设备、精度、预热/重复设置和查询规模下比较。

## 缓存和执行开关

生成缓存校验几何、查询网格、模型权重、结构、映射和数值设置；分析缓存校验质量及绘图选择。切换外部后处理通常可复用 raw 网络网格，切换模型结构或内部映射则必须重新前向。

| 开关 | 行为 |
| --- | --- |
| 无强制开关 | 复用身份一致且完整的生成和分析结果 |
| `--force-regenerate` | 重新生成网格 |
| `--force-reanalyze` | 复用网格并重算质量/图片 |
| `--force` | 两阶段均重做 |
| `--generation-only` | 只生成网格、边界误差和计时记录 |
| `--analysis-only` | 必须已有匹配网格，不执行模型或 PDE |
| `--no-plots` | 保留数值结果但不生成新图片 |
| `--quality-plots all` | 生成全部已启用的质量指标图和方法对比图 |
| `--quality-plots <metric> ...` | 只生成指定质量指标图；不影响所有指标的数值汇总 |
| `--save-distribution-plots` | 显式生成分布图 |
| `--no-compact-outputs` | 保留网格和逐单元中间数据 |

## 输出和保留策略

正式训练与评估根目录：

```text
outputs/
├── training/2D|3D/       # 日志、TensorBoard、训练清单
├── checkpoints/2D|3D/    # 阶段/全局最佳与最终模型
└── results/2D|3D/        # 正式评估
```

参数级评估目录按“维度/数据集/网格规模/参数”组织，包含配置身份、计时、边界误差、`results.json`、各方法目录和比较组。网络方法在目录内区分 raw、映射模式和外部后处理；比较目录名称包含实际参与的方法，不额外嵌套 PI-MeshONet 层级。

常见方法目录包含：

```text
mesh.npy / mesh_data.npz
mesh.png
mesh_metadata.json
grid_quality/
  quality_data.npz|csv
  quality_summary.json|csv
  invalid_cells.json|csv
  # 默认无质量图片；仅在 --quality-plots 时出现 quality_<metric>.png
boundary_error_summary.json
boundary_error_data.csv
```

比较组默认包含方法并排网格图、`results.json`/质量摘要、非法单元统计、边界误差和论文计时表；共享色标质量图仅在 `--quality-plots` 时生成。三维还可包含边界面与中间截面数值结果。

默认紧凑模式在汇总和图片完成后删除生成网格数组、逐单元 CSV/NPZ 和分析缓存，只长期保留汇总及启用的图片。需要下游求解、`analysis-only` 或自定义二次分析时，应使用 `--generation-only --no-compact-outputs` 保留网格；只有紧凑 `results.json` 时不能恢复原网格。

消融实验目录约定：

```text
experiments/<suite>/
├── baseline/<dataset>/...
├── <experiment>/<dataset>/...
├── eval/<dataset>/...
├── aggregate/<tag>/...
├── batch_runs/
└── queue_logs/<dataset>/<experiment>/seed_<seed>/<run-tag>/
```

多 GPU 队列把同一数据集、实验、随机种子和 run-tag 对应的训练与评估输出放在上述同一个目录中；训练指纹仅保留在队列 JSON 元数据中。完整流程使用 `train_evaluate.log`，只训练或只评估分别使用 `training.log`、`evaluation.log`；重复运行该任务时旧日志会在新子进程启动前清空。正式训练采用对应的 `outputs/gpu_queue/formal/<dataset>/<experiment>/seed_<seed>/<run-tag>/` 结构。正式多 GPU 的私有单模型评估会保留完整产物，并在任务全部完成后把方法级生成缓存发布到 `outputs/results/2D|3D/`；因此联合比较复用 HMONet、MeshONet、PI-MeshONet、TFI 与 PDE 的已有网格，仅补充跨方法汇总、对比图和联合质量表。

串行或多 GPU 批次结束后，主终端都会逐任务输出 `train=` 与 `evaluate=` 状态，以及失败任务的首个可读错误原因。消融实验的完整批次记录保存为 `experiments/<exN>/batch_runs/<run-tag>.json`；正式多 GPU 任务另保存 `outputs/gpu_queue/formal/completion_summary.json`。因失败未启动的队列任务会明确标记为 `skipped`，而不是从汇总中消失。

强制重训前应归档或删除同一运行目录的旧 TensorBoard event 文件，否则曲线会叠加。评估目录默认保留 `results.json` 和启用的网格、质量及对比图片；`--no-plots` 不会删除历史图片。

## 局部图

评估入口可以按物理坐标范围或最差质量单元生成局部图。独立脚本只读取现有非紧凑结果，不重新运行网络：

```bash
python scripts/evaluate/plot_result_zoom.py --parameter-dir <parameter-dir> --plot quality --metric scaled_jacobian --center-on worst
python scripts/evaluate/plot_result_zoom_3d.py --batch-root <parameter-dir> --mode slice --slice z_mid --metric scaled_jacobian --center-on worst
```

三维中间截面的物理投影约定为 x_mid→YZ、y_mid→XZ、z_mid→XY；偶数层数在中间两层插值到逻辑参数 0.5。

