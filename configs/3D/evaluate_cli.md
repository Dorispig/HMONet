# 三维评估命令行

三维评估入口为 `scripts/evaluate/evaluate_3d.py`，与二维使用同样的 `--case`、`--grids`、`--values` 规则。完整正式流程见 [三维训练—评估说明](train_evaluate_cli.md)。

## 统一简写

~~~powershell
python scripts/evaluate/evaluate_3d.py --case swb_a_ext_1_500_11_21_11 --methods tfi,pde,hmonet,meshonet,pimeshonet --grids 11x21x11 21x41x21 --values 0.02 0.34
~~~

每个显式 `--grids` 参数均按 `n_x×n_y×n_z` 书写：`--grids 11x31x11 21x61x21` 表示两个非立方网格。纯数字 `--grids 21 31` 自动展开为 `21×21×21`、`31×31×31`，适合立方尺度扫描。旧式逗号写法 `--grid-shapes 11x31x11,21x61x21` 仍兼容，且不能与 `--grids` 同时使用。

## 常见命令

~~~powershell
# 指定 HMONet checkpoint
python scripts/evaluate/evaluate_3d.py --case swb_a_ext_1_500_11_21_11 --methods tfi,pde,hmonet --hmonet-checkpoint outputs/checkpoints/3D/<checkpoint>/best.pt --grids 11x21x11 21x41x21 --values 0.02 0.34

# 强制重新评估
python scripts/evaluate/evaluate_3d.py --case swb_a_ext_1_500_11_21_11 --methods tfi,pde,hmonet,meshonet,pimeshonet --grids 11x21x11 21x41x21 --values 0.02 0.34 --force

# 输出指定质量图
python scripts/evaluate/evaluate_3d.py --case swb_a_ext_1_500_11_21_11 --methods tfi,pde,hmonet --grids 31 --values 0.02 --quality-plots minimum_scaled_jacobian center_condition_number

# 仅计算 z 中截面的二维四边形质量，附加最大角质量图
python scripts/evaluate/evaluate_3d.py --case swb_a_ext_1_500_11_21_11 --methods tfi,pde,hmonet --grids 11x21x11 --values 0.02 --middle-sections z_mid --quality-plots max_angle
~~~

| 参数 | 作用 |
|---|---|
| `--case` | 选择三维评估案例。 |
| `--methods` | 选择 TFI、PDE 和网络。 |
| `--grids` | 输出网格形状；每个形状可显式写为 `n_x × n_y × n_z`，纯数字时才为立方边长。 |
| `--values` | 物理参数值。 |
| `--quality-metrics` | `both`、`none` 或逗号分隔的质量指标。 |
| `--quality-plots` | `all` 或一个以上质量图名称。 |
| `--middle-sections` | 默认 `all`。可写 `none` 关闭切片，或指定一个或多个截面，例如 `x_mid z_mid`。 |
| `--force` | 重新生成和分析。 |

中截面分别固定逻辑坐标 `ξ=0.5`、`η=0.5`、`ζ=0.5`，偶数层取中间两层插值。截面指标按物理空间中四边形的几何量计算，图片投影到对应的坐标平面。截面数值汇总保存在 `results.json` 的 `middle_section_metrics`。体网格指标独立计算，共用的指标名（例如 `max_angle`）会分别用于体网格和选中的截面。用于局部放大的 `--slice` 是单独的参数。

三维网格成本按立方增长，应根据显存和后处理成本选择 `--grids`。模型选择与二维一致。

~~~powershell
python scripts/evaluate/evaluate_3d.py --help
~~~
