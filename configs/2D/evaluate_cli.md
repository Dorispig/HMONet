# 二维评估命令行

二维评估入口为 `scripts/evaluate/evaluate_2d.py`。训练后优先使用串联入口自动评估；本入口用于重新分析或评估已有 checkpoint。完整正式流程见 [二维训练—评估说明](train_evaluate_cli.md)。

## 统一简写

~~~powershell
python scripts/evaluate/evaluate_2d.py --case air_t_ext_1_500_41_41 --methods tfi,pde,hmonet,meshonet,pimeshonet --grids 101 201 401 --values 0.62 0.65
~~~

`--grids 101 201 401` 自动展开为 `101×101`、`201×201`、`401×401`。旧式 `--grid-shapes 101x101,201x201,401x401` 与逗号参数值仍兼容。

## 常见命令

~~~powershell
# 指定 HMONet checkpoint
python scripts/evaluate/evaluate_2d.py --case air_t_ext_1_500_41_41 --methods tfi,pde,hmonet --hmonet-checkpoint outputs/checkpoints/2D/<checkpoint>/best.pt --grids 101 201 --values 0.62 0.65

# 强制重新评估
python scripts/evaluate/evaluate_2d.py --case air_t_ext_1_500_41_41 --methods tfi,pde,hmonet,meshonet,pimeshonet --grids 101 201 401 --values 0.62 0.65 --force

# 生成指定质量图
python scripts/evaluate/evaluate_2d.py --case air_t_ext_1_500_41_41 --methods tfi,pde,hmonet --grids 101 --values 0.62 --quality-plots scaled_jacobian max_angle
~~~

| 参数 | 作用 |
|---|---|
| `--case` | 选择 evaluate.yaml 中的案例。 |
| `--methods` | 逗号分隔的比较对象。 |
| `--grids` | 方形输出网格边长。 |
| `--values` | 物理参数值。 |
| `--quality-metrics` | `both`、`none` 或逗号分隔指标。 |
| `--quality-plots` | `all` 或一个以上质量图名称。 |
| `--force` | 重新生成和分析。 |
| `--analysis-only` | 仅使用已有匹配网格做分析。 |
| `--generation-only` | 仅生成网格。 |

默认保存网格图和数值汇总，不保存逐质量指标图；使用 `--quality-plots all` 输出全部质量图。模型选择与训练入口一致。

~~~powershell
python scripts/evaluate/evaluate_2d.py --help
~~~
