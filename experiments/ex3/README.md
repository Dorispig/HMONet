# HMONet网络架构消融

基线使用CCAP精确硬边界映射、幂次—Fourier坐标增强和归一化复合物理目标。本套件逐项测试：

| 名称 | 相对基线的修改 |
| --- | --- |
| `e1_modified_mlp` | 分支和主干改为门控ModifiedMLP |
| `e2_cross_attention` | 增加查询点到边界的交叉注意力 |
| `e3_boundary_decoder` | 增加边界层专用解码器 |
| `e4_decoupled_weights` | 解耦原始坐标与映射权重路径 |
| `e5_qg_hmonet` | Query–Geometry Gated HMONet |
| `e6_modified_cross` | ModifiedMLP+交叉注意力 |
| `e7_modified_cross_boundary` | 再增加边界层解码器 |
| `e8_full_factorized` | 再增加权重解耦 |
| `e9_meshonet_200k` | 约20万参数MeshONet |
| `e10_pimeshonet_200k` | 约20万参数PI-MeshONet |
| `e11_raw_pde` | 将归一化PDE替换为原始PDE残差 |

默认数据集为 `air_t_ext_1_500_21_21`。

```bash
python scripts/experiments/run.py ex3 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1
```

QG-HMONet不显式计算曲率、法向或SDF；它使用多尺度边界编码、查询—边界交叉注意力和门控融合，同时保留精确硬边界映射。

