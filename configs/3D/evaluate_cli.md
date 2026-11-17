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
~~~

| 参数 | 作用 |
|---|---|
| `--case` | 选择三维评估案例。 |
| `--methods` | 选择 TFI、PDE 和网络。 |
| `--grids` | 输出网格形状；每个形状可显式写为 `n_x × n_y × n_z`，纯数字时才为立方边长。 |
| `--values` | 物理参数值。 |
| `--quality-metrics` | `both`、`none` 或逗号分隔的质量指标。 |
| `--quality-plots` | `all` 或一个以上质量图名称。 |
| `--force` | 重新生成和分析。 |

三维网格成本按立方增长，应根据显存和后处理成本选择 `--grids`。模型选择与二维一致。

~~~powershell
python scripts/evaluate/evaluate_3d.py --help
~~~
