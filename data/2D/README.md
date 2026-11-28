# 2D 数据集说明与命令

公式说明集中在本目录的 `boundaries/` 中；生成的图片仍放在 `outputs/boundaries/`，公式文件内使用相对链接引用。

- [九类默认参数的生成命令（外推与内插）](generation_commands.md)
- [正式训练与评估命令](../../configs/2D/formal_external_commands.md)

| 缩写与公式 | 边界名称 | 可变参数 | 默认节点规模 |
|---|---|---|---|
| [air_t](boundaries/air_t.md) | `circle_airfoil_t` | `t` | 41×41 |
| [esa](boundaries/esa.md) | `eccentric_semiannulus` | `inner_center_x` | 41×41 |
| [rse](boundaries/rse.md) | `rect_top_semiellipse` | `semi_minor_axis` | 41×41 |
| [run](boundaries/run.md) | `rounded_u_notch_compact` | `depth` | 41×41 |
| [scc](boundaries/scc.md) | `smooth_constricted_channel` | `depth` | 41×41 |
| [vsc](boundaries/vsc.md) | `variable_width_s_channel` | `constriction` | 41×41 |
| [tri_p](boundaries/tri_p.md) | `symmetric_convex_tri` | `protrusion_distance` | 41×41 |
| [gsp_a](boundaries/gsp_a.md) | `gear_spline` | `amplitude` | 41×41 |
| [fca_a](boundaries/fca_a.md) | `four_convex_arcs` | `arc_angle_deg` | 41×41 |

## 生成后运行正式实验

数据集可用于默认三个模型和 ResMetaMesh。ResMetaMesh 用训练集样本索引关联潜码，复用数据集的参数边界，预训练无需 PDE 坐标标签。完整批量训练和评估命令见 [2D 临时命令](../../configs/2D/formal_external_commands.md#一次性训练并评估全部九个数据集)。
