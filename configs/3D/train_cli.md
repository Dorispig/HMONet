# 三维训练命令行说明

入口为 `scripts/train/train_3d.py`，默认配置为 `configs/3D/train.yaml`。本文只说明三维特有内容；公共的收敛、优化器和日志参数与[二维训练说明](../2D/train_cli.md)一致。

## 常用命令

按 YAML 训练：

```bash
python scripts/train/train_3d.py --config configs/3D/train.yaml
```

只训练外推 HMONet：

```bash
python scripts/train/train_3d.py --config configs/3D/train.yaml --model hmonet --dataset-mode extrapolation --device cuda
```

训练 HMONet、MeshONet 和 PI-MeshONet：

```bash
python scripts/train/train_3d.py --model hmonet,meshonet,pimeshonet --dataset-mode both
```

显式指定一个数据集并检查其训练网格：

```bash
python scripts/train/train_3d.py --dataset data/raw/3D/sch_r_ext_0_1000_11_31_11 --dataset-mode extrapolation --model hmonet --nx 11 --ny 31 --nz 11
```

## 三维专有参数

- `--nx/--ny/--nz`：只校验数据集内的训练点数，不重采样数据集。
- 正式 HMONet 配置与二维一致，使用 `200` 个全局均匀点和 `200` 个曲率自适应点；三维自适应点通过“曲率加权参数面锚点 + 向域内局部扰动”生成，避免构造候选点到全部面点的巨大距离矩阵。
- 正式 HMONet 从头采用归一化椭圆网格 PDE、归一化 Jacobian 屏障和最小奇异值屏障进行自监督训练，不读取 PDE 网格作为坐标监督目标。
- `--pi-fd-step HX HY HZ`：PI 模型三个计算坐标方向的有限差分步长。
- `--hmonet-mapping-mode`：正式训练默认 `corner_corrected_affine`（CCAP），另可选 `rational_affine` 或 `coons_residual`。
- `--boundary-self-attention-enabled`：启用三维六面联合边界注意力编码。

`--sample-count`、`--jacobian-fd-step` 和 `--target-interpolation` 仅用于兼容旧的监督式 HMONet 配置，不改变当前正式自监督配置中的 `random_loss_*` 与 `pde_fd_step`。

模型别名：`both` 为 HMONet 与 MeshONet，`pi` 为三个 PI 模型，`all` 为五种神经网络；也可以传逗号分隔的任意列表。

训练规模始终从 `dataset.npy` 的 `grid_shape` 读取。更换 `11×31×11`、`21×21×21` 等训练规模时，应切换到对应数据集，不能仅修改 `--nx/--ny/--nz`。

## 输出

默认训练记录位于 `outputs/training/3D/`，检查点位于 `outputs/checkpoints/3D/`。每个模型、数据集和随机种子使用独立目录。TensorBoard、输出保留和检查点选择规则见[评估体系与输出结构](../../docs/evaluation_system.md)及[复现与兼容性说明](../../docs/reproducibility.md)。

查看当前版本的完整参数：

```bash
python scripts/train/train_3d.py --help
```

