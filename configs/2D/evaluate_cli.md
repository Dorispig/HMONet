# 二维评估命令行

二维评估入口为 `scripts/evaluate/evaluate_2d.py`。训练后优先使用串联入口自动评估；本入口用于重新分析或评估已有 checkpoint。完整正式流程见 [二维训练—评估说明](train_evaluate_cli.md)。

## 统一简写

~~~powershell
python scripts/evaluate/evaluate_2d.py --case air_t_ext_1_500_21_21 --methods tfi,pde,hmonet,meshonet,pimeshonet --grids 101 201 401 --values 0.62 0.65
~~~

`--grids 101 201 401` 自动展开为 `101×101`、`201×201`、`401×401`。旧式 `--grid-shapes 101x101,201x201,401x401` 与逗号参数值仍兼容。

## 常见命令

~~~powershell
# 指定 HMONet checkpoint
python scripts/evaluate/evaluate_2d.py --case air_t_ext_1_500_21_21 --methods tfi,pde,hmonet --hmonet-checkpoint outputs/checkpoints/2D/<checkpoint>/best.pt --grids 101 201 --values 0.62 0.65

# 强制重新评估
python scripts/evaluate/evaluate_2d.py --case air_t_ext_1_500_21_21 --methods tfi,pde,hmonet,meshonet,pimeshonet --grids 101 201 401 --values 0.62 0.65 --force

# 生成指定质量图
python scripts/evaluate/evaluate_2d.py --case air_t_ext_1_500_21_21 --methods tfi,pde,hmonet --grids 101 --values 0.62 --quality-plots scaled_jacobian max_angle
~~~

| 参数 | 作用 |
|---|---|
| `--case` | 选择 evaluate.yaml 中的案例。 |
| `--methods` | 逗号分隔的比较对象。 |
| `--grids` | 方形输出网格边长。 |
| `--values` | 物理参数值。 |
| `--quality-metrics` | `both`、`none` 或逗号分隔指标。 |
| `--quality-plots` | `all` 或一个以上质量图名称。 |
| `--exact-boundary-overlay` / `--no-exact-boundary-overlay` | 默认开启：在保留普通对比图的同时，额外生成红色精确连续边界叠加图；关闭后仅生成普通图。只影响图片。 |
| `--force` | 重新生成和分析。 |
| `--analysis-only` | 仅使用已有匹配网格做分析。 |
| `--generation-only` | 仅生成网格。 |

正式结果保存在 `outputs/results/2D/<数据集>/<网格>/<参数>/`。单方法结果按模型和 `raw`、`idw`、`direct` 分开存放，HMONet 再按硬约束方式区分。联合结果位于 `comparisons/`，每个对比组保存自己的质量数据和数值汇总。同一对比组重新评估时覆盖同名图；不同后处理方式各自保留。

默认保存网格图和数值汇总，不保存逐质量指标图；使用 `--quality-plots all` 输出全部质量图。默认额外生成 `mesh_comparison_with_exact_boundary.png`，原始 `mesh_comparison_overview.png` 不会被覆盖。模型选择与训练入口一致。

~~~powershell
python scripts/evaluate/evaluate_2d.py --help
~~~

## 数据集级论文对比图

已有五方法正式评估后，可一次生成三模型训练时间/最佳 MSE 柱状图、五方法推理时间图，以及每个参数、网格和质量指标的六张统计对比图：

~~~powershell
python scripts/evaluate/generate_dataset_comparison_figures.py --dimension 2 --dataset air_t_ext_1_500_21_21 --grids 101x101 201x201 401x401 --values 0.62 0.65 --quality-metrics all
~~~

每项质量指标固定输出六张五方法对比图：`raw/direct/IDW × mean/worst`。每张图的顺序固定为 TFI、PDE、MeshONet、PI-MeshONet、HMONet；前三种精确边界结果（TFI、PDE、HMONet）在三类图中保持不变，两个对比模型使用对应的 raw、direct 或 IDW 结果。推理时间图也采用该方法顺序；MeshONet 与 PI-MeshONet 在单个横轴位置以不同宽度重叠显示 raw、raw+direct、raw+IDW 三种总耗时。训练图位于 `outputs/results/2D/<数据集>/dataset_comparisons/training/`；其余图位于各参数目录的 `summary_comparisons/`。脚本从已有 raw 网格派生 direct/IDW，不重新运行网络。

除网格几何质量外，脚本固定增加 `mse_vs_pde` 六图。其 mean 是包含边界和内部点的所有网格点坐标分量MSE；worst 是单个网格点的最大坐标分量MSE。两者均以同一参数、同一规模的PDE网格为参考，PDE自身为0。

连续质量指标的五个结果若集中在很窄的正值区间，纵轴会自动聚焦并保留边距，使细微差异可读；差距较大、包含零值或属于非法单元计数的指标仍保留零基线，避免夸大差异。
