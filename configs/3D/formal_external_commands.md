# 三维六类外推正式实验：临时命令清单

本文件用于逐个训练并评估六类三维外推数据集。每条命令均依次完成 HMONet、MeshONet、PI-MeshONet 的训练与评估。默认列出 `11×21×11`、`21×41×21`、`41×81×41` 三个网格规模；第三个规模开销较高，可先从命令中删除，训练完成后再单独评估。默认不使用外部边界后处理；三种网络的未处理输出统一存入 `raw/`。IDW 仅作为额外对照，通过命令行显式指定。

在项目根目录 `E:\projects\HMONet` 运行。运行前请确认对应数据集已生成；数据集生成命令见 [data/3D/generation_commands.md](../../data/3D/generation_commands.md)。

三维评估默认计算三个逻辑中截面。论文只展示 `z_mid` 时，可在任一命令末尾加上 `--middle-sections z_mid`。只评估体网格时使用 `--middle-sections none`。

```powershell
# cylindrical_domain_with_airfoil_hole (cah_t)
python scripts/train/train_evaluate.py --dimension 3 --dataset cah_t_ext_1_500_11_21_11 --models hmonet meshonet pimeshonet --grids 11x21x11 21x41x21 41x81x41 --values 0.06 0.38 --seed 20261030 --run-tag v1

# sidewall_deformed_block (swb_a)
python scripts/train/train_evaluate.py --dimension 3 --dataset swb_a_ext_1_500_11_21_11 --models hmonet meshonet pimeshonet --grids 11x21x11 21x41x21 41x81x41 --values 0.02 0.34 --seed 20261030 --run-tag v1

# rounded_u_notch_extrusion (run_e): exact extrusion of the 2-D run cross-section
python scripts/train/train_evaluate.py --dimension 3 --dataset run_e_ext_1_500_41_41_11 --models hmonet meshonet pimeshonet --grids 41x41x11 81x81x21 161x161x41 --values 1.52 1.55 --seed 20261030 --run-tag v1

# tapered_elliptic_bore_block (teb_b)
python scripts/train/train_evaluate.py --dimension 3 --dataset teb_b_ext_1_500_11_21_11 --models hmonet meshonet pimeshonet --grids 11x21x11 21x41x21 41x81x41 --values 0.08 0.36 --seed 20261030 --run-tag v1

# bent_annular_duct (bad_a)
python scripts/train/train_evaluate.py --dimension 3 --dataset bad_a_ext_1_500_11_21_11 --models hmonet meshonet pimeshonet --grids 11x21x11 21x41x21 41x81x41 --values 10 105 --seed 20261030 --run-tag v1

# lobed_annular_duct (lad_a)
python scripts/train/train_evaluate.py --dimension 3 --dataset lad_a_ext_1_500_11_21_11 --models hmonet meshonet pimeshonet --grids 11x21x11 21x41x21 41x81x41 --values 0 0.20 --seed 20261030 --run-tag v1
```

服务器多 GPU 队列：将任一命令末尾追加 `--gpus 0 1 2 3`。同一数据集的 HMONet、MeshONet、PI-MeshONet 会作为三个独立的训练—评估任务并行分配至可用 GPU；各任务保存自身、TFI、PDE 的评估产物，全部结束后联合阶段复用这些网格，仅生成统一的多模型质量汇总和对比图。
