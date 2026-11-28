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

## 六方法网格与质量图

已保存的 raw 网格可直接生成六方法对比图。以下命令读取与检查点和评估设置匹配的已有网格：

~~~powershell
python scripts/train/train_evaluate.py --dimension 2 --dataset air_t_ext_1_500_41_41 --models hmonet meshonet pimeshonet resmetamesh --evaluate-only --analysis-only --evaluate-methods tfi,pde,meshonet,pimeshonet,hmonet,resmetamesh --grids 101 201 401 --values 0.62 0.65 --device cuda:0
~~~

输出位于 `outputs/results/2D/<数据集>/<网格>/<参数>/comparisons/CCAP/H_raw__M_raw/TFI_PDE_M_PI_H_RM/`。普通网格图为 `mesh_comparison_overview.png`，精确边界叠加图为 `mesh_comparison_with_exact_boundary.png`。追加 `--quality-plots scaled_jacobian max_angle` 可以生成六方法质量分布图，追加 `--quality-plots all` 可以生成所有质量分布图。其他数据集使用各自的评估参数。已有 direct 和 IDW 六方法网格图分别位于 `comparisons/CCAP/H_raw__M_direct__PI_direct/TFI_PDE_M_PI_H_RM/` 与 `comparisons/CCAP/H_raw__M_idw__PI_idw/TFI_PDE_M_PI_H_RM/`。这两组中只有 MeshONet 和 PI-MeshONet 使用对应边界后处理。

## 数据集级柱状对比图

已完成六方法评估的数据集可直接复用现有统计量和耗时生成柱状图：

~~~powershell
python scripts/evaluate/generate_dataset_comparison_figures.py --dimension 2 --dataset esa_ext_1_500_41_41 --include-resmetamesh --skip-training-comparison
~~~

默认指标为 `max_angle`。不指定网格和参数时遍历该数据集的全部已评估工况，也可以用 `--grids 101x101 201x201` 和 `--values 0.72 0.75` 筛选。指定 `--quality-metrics all` 会输出全部网格质量指标及 `mse_vs_pde`。也可使用 `--quality-metrics max_angle scaled_jacobian` 只生成指定指标。

每个工况保存到 `outputs/results/2D/<数据集>/<网格>/<参数>/summary_comparisons/six_methods/`：

- `quality/<指标>/` 每个指标有六张图，分别为 raw、direct、IDW 的均值和最差值。顺序为 TFI、PDE、MeshONet、PI-MeshONet、HMONet、ResMetaMesh。仅 MeshONet 和 PI-MeshONet 随组切换边界处理。
- `inference_time_bar.png` 是六方法总生成时间图。MeshONet、PI-MeshONet 在同一位置分层展示 raw、raw + direct 和 raw + IDW 的实测总时间，ResMetaMesh 的耗时含逐几何适配。
- `inference_time.csv` 和 `quality_statistics.csv` 保存柱状图对应数值。几何质量和时间来自已保存的评估记录，绘图阶段不运行网络、PDE 求解或边界后处理。
- `mse_vs_pde` 的 mean 是所有网格节点的坐标分量 MSE，worst 是最大单节点坐标分量 MSE。TFI 对 PDE 的 MSE 从已保存的两套坐标计算。

对有六方法 raw 评估的工况，脚本自动使用六方法模式。`--include-resmetamesh` 要求选定工况都有六方法评估，缺失时给出文件路径。原五方法柱状图保留在父级 `summary_comparisons/`。

## ResMetaMesh 单独评估与结果位置

仅评估 ResMetaMesh 时，显式指定 `--evaluate-methods resmetamesh`：

```bash
python scripts/train/train_evaluate.py --dimension 2 --dataset air_t_ext_1_500_41_41 --models resmetamesh --evaluate-only --evaluate-methods resmetamesh
```

网格、质量、边界误差、计时和 `evaluation_result.json` 集中写入该工况的 `ResMetaMesh/raw/`。单方法评估不会创建 `comparisons/RM/`，也不会刷新工况顶层的联合结果。已保存的单方法网格可以在后续联合评估时复用。

同时生成六方法联合对比：


```bash
python scripts/train/train_evaluate.py --dimension 2 --dataset air_t_ext_1_500_41_41 --models resmetamesh --evaluate-only --evaluate-methods tfi,pde,meshonet,pimeshonet,hmonet,resmetamesh
```

六方法联合评估需要该数据集上其余三个学习模型已有兼容 checkpoint。若六种网格均已评估，再加 `--analysis-only` 可直接读取兼容网格和缓存。ResMetaMesh 使用 `final.pt`，其余模型默认使用 `best.pt`。原始网格和单方法质量位于参数目录的 `ResMetaMesh/raw/`，六方法联合目录以 `TFI_PDE_M_PI_H_RM` 命名。ResMetaMesh 的端到端生成用时包含逐几何适配。
