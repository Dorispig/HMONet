# 3D 数据集说明与命令

公式说明集中在本目录的 `boundaries/` 中；生成的图片仍放在 `outputs/boundaries/`，公式文件内使用相对链接引用。

- [九类默认参数的生成命令（外推与内插）](generation_commands.md)
- [正式训练与评估命令](../../configs/3D/formal_external_commands.md)

| 缩写与公式 | 边界名称 | 可变参数 | 默认节点规模 |
|---|---|---|---|
| [cah_t](boundaries/cah_t.md) | `cylindrical_domain_with_airfoil_hole` | `thickness` | 21×21×11 |
| [swb_a](boundaries/swb_a.md) | `sidewall_deformed_block` | `sidewall_amplitude` | 21×21×11 |
| [run_e](boundaries/run_e.md) | `rounded_u_notch_extrusion` | `depth` | 41×41×11 |
| [teb_b](boundaries/teb_b.md) | `tapered_elliptic_bore_block` | `minor_axis_out` | 21×21×11 |
| [bad_a](boundaries/bad_a.md) | `bent_annular_duct` | `bend_angle_deg` | 21×21×11 |
| [lad_a](boundaries/lad_a.md) | `lobed_annular_duct` | `amplitude` | 21×21×11 |
| [srt_r](boundaries/srt_r.md) | `square_to_round_transition` | `outlet_radius` | 21×21×11 |
| [sad_c](boundaries/sad_c.md) | `variable_area_s_duct` | `area_ratio_out` | 21×21×11 |
| [tbp_a](boundaries/tbp_a.md) | `twisted_blade_passage` | `twist_angle_deg` | 21×21×11 |

三维数据生成默认使用CPU，也可通过 `--solver-backend cuda --gpus 0 1` 启用数据集级多GPU生成；完整参数和示例见[三维生成说明](generation_commands.md)。训练GPU参数与数据生成GPU参数分别设置。

三维 GPU 数据生成采用可选 cuDSS float64 稀疏直接求解，与 CPU 共用矩阵组装和外层迭代。cuDSS 已纳入总依赖，统一安装 `python -m pip install -r requirements.txt`；CPU 仍为默认，运行时不调用 cuDSS。完整 CPU/多 GPU 命令见 [生成命令](generation_commands.md)。原 `--linear-tolerance`、`--linear-max-iter` 已移除。

GPU 以完整数据集分配任务；详细进度按前几个样本耗时和每 10% 完成度写入 `outputs/gpu_queue/generation/3D/<数据集名称>/generation.log`。可用 `--queue-log-root` 修改日志根目录，终端仅显示任务概况和错误。
