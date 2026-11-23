# 三维评估命令行

三维评估入口为 `scripts/evaluate/evaluate_3d.py`，与二维使用同样的 `--case`、`--grids`、`--values` 规则。完整正式流程见 [三维训练—评估说明](train_evaluate_cli.md)。

当前九类边界使用新的 PDE 端面参数化。仅评估时也会核对 checkpoint 中的 `dataset_geometry`（边界族、几何修订号、端面和侧面方案）。旧参数面训练的模型不能直接用于新参数面；需要重新生成数据并重新训练三个模型，不能通过重命名权重绕过。三维数据集名称默认版本为 2，内部几何修订号仍为 3；实验标签 `--run-tag v1` 与数据集版本相互独立。

## 统一简写

~~~powershell
python scripts/evaluate/evaluate_3d.py --case swb_a_ext_2_500_21_21_11 --methods tfi,pde,hmonet,meshonet,pimeshonet --grids 41x41x11 81x81x11 --values 0.02 0.34
~~~

每个显式 `--grids` 参数均按 `n_x×n_y×n_z` 书写：`--grids 11x31x11 21x61x21` 表示两个非立方网格。纯数字 `--grids 21 31` 自动展开为 `21×21×21`、`31×31×31`，适合立方尺度扫描。旧式逗号写法 `--grid-shapes 11x31x11,21x61x21` 仍兼容，且不能与 `--grids` 同时使用。

## 常见命令

~~~powershell
# 指定 HMONet checkpoint
python scripts/evaluate/evaluate_3d.py --case swb_a_ext_2_500_21_21_11 --methods tfi,pde,hmonet --hmonet-checkpoint outputs/checkpoints/3D/<checkpoint>/best.pt --grids 41x41x11 81x81x11 --values 0.02 0.34

# 强制重新评估
python scripts/evaluate/evaluate_3d.py --case swb_a_ext_2_500_21_21_11 --methods tfi,pde,hmonet,meshonet,pimeshonet --grids 41x41x11 81x81x11 --values 0.02 0.34 --force

# 输出指定质量图
python scripts/evaluate/evaluate_3d.py --case swb_a_ext_2_500_21_21_11 --methods tfi,pde,hmonet --grids 31 --values 0.02 --quality-plots minimum_scaled_jacobian center_condition_number

# 仅计算 z 中截面的二维四边形质量，附加最大角质量图
python scripts/evaluate/evaluate_3d.py --case swb_a_ext_2_500_21_21_11 --methods tfi,pde,hmonet --grids 41x41x11 81x81x11 --values 0.02 --no-volume-evaluation --middle-sections z_mid --quality-plots max_angle
~~~

| 参数 | 作用 |
|---|---|
| `--case` | 选择三维评估案例。 |
| `--methods` | 选择 TFI、PDE 和网络。 |
| `--grids` | 输出网格形状；每个形状可显式写为 `n_x × n_y × n_z`，纯数字时才为立方边长。 |
| `--values` | 物理参数值。 |
| `--quality-metrics` | `both`、`none` 或逗号分隔的质量指标。 |
| `--quality-plots` | `all` 或一个以上质量图名称。 |
| `--volume-evaluation` / `--no-volume-evaluation` | 默认开启体网格质量及体网格图；关闭时不调用六面体质量计算。仍生成/复用完整坐标网格，供截面提取和后续复用。 |
| `--middle-sections` | 默认 `z_mid`。可写 `none` 关闭截面，`all` 开启三个截面，或指定任意组合，例如 `x_mid z_mid`。独立于体网格开关。 |
| `--exact-boundary-overlay` / `--no-exact-boundary-overlay` | 默认开启：保留普通截面图，并额外生成带红色精确连续边界的截面图；关闭后仅生成普通截面图。体网格图不叠加边界。 |
| `--force` | 重新生成和分析。 |

中截面分别固定逻辑坐标 `ξ=0.5`、`η=0.5`、`ζ=0.5`，偶数层取中间两层插值。截面指标按物理空间中四边形的几何量计算，图片投影到对应的坐标平面。截面数值汇总保存在 `results.json` 的 `middle_section_metrics`。体网格指标独立计算，共用的指标名（例如 `max_angle`）会分别用于体网格和选中的截面。用于局部放大的 `--slice` 是单独的参数。

默认每个选中截面同时保存 `middle_section_mesh_overview.png` 和 `middle_section_mesh_with_exact_boundary.png`。后者仅增加绘图曲线，不新增网格、质量或边界误差计算。

四个评估部分可以独立组合（下方参数也可直接用于正式 `train_evaluate.py` 入口，包括多 GPU）：

```powershell
# 默认：体网格 + z_mid
python scripts/evaluate/evaluate_3d.py --case cah_t_ext_2_500_21_21_11 --methods tfi,pde --grids 41x41x11 81x81x11 --volume-evaluation --middle-sections z_mid
# 只算 z_mid，跳过昂贵的体网格质量计算
python scripts/evaluate/evaluate_3d.py --case cah_t_ext_2_500_21_21_11 --methods tfi,pde --grids 41x81x41 --no-volume-evaluation --middle-sections z_mid
# 只算体网格：--volume-evaluation --middle-sections none
# 只算 x_mid 与 y_mid：--no-volume-evaluation --middle-sections x_mid y_mid
# 四项全开：--volume-evaluation --middle-sections all
# 四项全关：--no-volume-evaluation --middle-sections none
```

关闭全部质量区域仍保留完整网格及已有的计时、边界误差等公共结果。关闭体网格评估不能省略三维坐标预测/PDE 求解，因为截面仍从同一个体网格提取。评估区域会进入分析指纹；切换区域不会把旧质量摘要误认为当前请求已完成。已有匹配网格可用 `--analysis-only` 复用，新增区域仍需实际计算质量。

正式结果保存在 `outputs/results/3D/<数据集>/<网格>/<参数>/`。例如 `comparisons/CCAP/H_raw__M_raw/TFI_PDE_M_PI_H/` 下保存联合对比，`boundary_faces/` 保存中截面结果，`grid_quality/` 保存体网格质量。每个对比组保留自己的数值汇总，同组重新评估时覆盖同名图。三维完整评估成功后清理该案例的 `.cache`，后续可通过正式方法目录中的网格和生成记录复用计算。

三维网格成本按立方增长，应根据显存和后处理成本选择 `--grids`。模型选择与二维一致。

### 质量计算加速

六面体质量采用 NumPy 批量计算，每批最多 32768 个单元，避免逐单元 Python 循环和一次性展开全部体网格；只计算请求指标需要的中间量。截面四边形的角度、Jacobian、边长及局部相交检测也采用批量计算。指标定义、单元顺序和非法单元判据保持不变，不影响已有数值结果的复用。

本地 CPU 的合成变形网格对照测试：`41×81×41` 体网格全指标由约 51.52 秒降至 0.685 秒；`201×201` 截面全指标由约 6.35 秒降至 0.067 秒。新实现预热后取三次平均，旧实现测一次，逐指标核对数值及翻转/退化判定。这些是质量计算耗时，不包含网格生成、全局 IDW、绘图和文件读写，不能当作端到端评估加速倍数。

~~~powershell
python scripts/evaluate/evaluate_3d.py --help
~~~

## 数据集级论文对比图

三维使用同一脚本和目录规则；`--grids` 保持 `n_x×n_y×n_z`：

~~~powershell
python scripts/evaluate/generate_dataset_comparison_figures.py --dimension 3 --dataset swb_a_ext_2_500_21_21_11 --grids 41x41x11 81x81x11 --values 0.02 0.34 --quality-metrics all --idw-chunk-size 256
~~~

每个体网格质量指标同样输出六张五方法柱状图：`raw/direct/IDW × mean/worst`，顺序固定为 TFI、PDE、MeshONet、PI-MeshONet、HMONet。推理时间图中 MeshONet 与 PI-MeshONet 在单个横轴位置重叠显示 raw、raw+direct、raw+IDW 三种总耗时。三维全局 IDW 需要计算所有内部点到边界面的距离，计算量明显高于二维；可通过 `--idw-chunk-size` 控制内存，通过 `--timing-repeats` 控制计时重复次数。输出位于 `outputs/results/3D/<数据集>/<网格>/<参数>/summary_comparisons/`。

脚本同时增加以同规模PDE网格为参考的 `mse_vs_pde` 六图：mean为所有网格点坐标分量MSE，worst为最大单点坐标分量MSE。

连续质量指标若集中在很窄的正值区间，纵轴会自动聚焦并保留边距；零值计数和跨度较大的指标仍保留零基线。
