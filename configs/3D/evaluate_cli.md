# 三维评估命令行说明

入口为 `scripts/evaluate/evaluate_3d.py`，默认读取 `configs/3D/evaluate.yaml` 和 `configs/3D/train.yaml`。

## 常用命令

评估一个配置案例：

```bash
python scripts/evaluate/evaluate_3d.py --case sch_r_ext_0_1000_11_31_11 --methods tfi,pde,meshonet,pimeshonet,hmonet --grid-shapes 21x21x21,31x31x31
```

指定物理参数和 checkpoint：

```bash
python scripts/evaluate/evaluate_3d.py --case sch_r_ext_0_1000_11_31_11 --values 0.2,0.3 --grid-shapes 21x21x21 --hmonet-checkpoint outputs/checkpoints/3D/<checkpoint>.pt
```

只生成网格和图片、不计算质量：

```bash
python scripts/evaluate/evaluate_3d.py --case sch_r_ext_0_1000_11_31_11 --grid-shapes 21x21x21 --quality-metrics none
```

保留 `generated_meshes.npz` 供下游分析：

```bash
python scripts/evaluate/evaluate_3d.py --case sch_r_ext_0_1000_11_31_11 --grid-shapes 21x21x21 --generation-only --no-compact-outputs
```

## 三维网格质量

三维体网格可选择 `minimum_scaled_jacobian`、`center_jacobian`、`center_condition_number`、`edge_aspect_ratio`、`cell_volume`、`flipped_cell` 等六面体指标；中间截面沿用二维四边形指标。项目默认不把 `cell_area` 作为正式对比指标。用 `--quality-metrics both` 选择配置允许的全部指标，或传逗号分隔列表。

局部质量图使用：

```bash
python scripts/evaluate/evaluate_3d.py --case sch_r_ext_0_1000_11_31_11 --analysis-only --metric minimum_scaled_jacobian --center-on worst --zoom-mode slice --slice z_mid
```

`--slice` 可选 `x_mid`、`y_mid`、`z_mid`；`--zoom-mode volume` 用于三维体局部范围，并可通过 `--xlim`、`--ylim`、`--zlim` 限定物理坐标。

## 边界与缓存

- 五种神经网络的边界后处理默认均为 `none`；只有显式传入 `--<model>-boundary-projection` 才改变输出。
- `--compact-outputs` 默认只长期保留 `results.json` 与图片；需要网格数组时使用 `--generation-only` 或 `--no-compact-outputs`。
- `--force-regenerate` 只重新生成网格，`--force-reanalyze` 只重算质量，`--force` 同时强制两者。
- `--analysis-only` 要求已有匹配的网格缓存；紧凑结果已删除网格时需要先重新生成。

查看完整参数：

```bash
python scripts/evaluate/evaluate_3d.py --help
```

