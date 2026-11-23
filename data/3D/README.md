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
