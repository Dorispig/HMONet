# 三维六类外推正式实验：临时命令清单

本文件用于逐个训练并评估六类三维外推数据集。每条命令均依次完成 HMONet、MeshONet、PI-MeshONet 的训练与评估。默认列出 `11×21×11`、`21×41×21`、`41×81×41` 三个网格规模；第三个规模开销较高，可先从命令中删除，训练完成后再单独评估。

在项目根目录 `E:\projects\HMONet` 运行。运行前请确认对应数据集已生成；数据集生成命令见 [data/3D/generation_commands.md](../../data/3D/generation_commands.md)。

```powershell
# cylindrical domain with airfoil hole
python scripts/train/train_evaluate.py --dimension 3 --dataset cah_t_ext_1_500_11_21_11 --models hmonet meshonet pimeshonet --grids 11x21x11 21x41x21 41x81x41 --values 0.06 0.38 --seed 20261030 --run-tag v1

# sidewall-deformed block
python scripts/train/train_evaluate.py --dimension 3 --dataset swb_a_ext_1_500_11_21_11 --models hmonet meshonet pimeshonet --grids 11x21x11 21x41x21 41x81x41 --values 0.02 0.34 --seed 20261030 --run-tag v1

# rounded U-groove extrusion
python scripts/train/train_evaluate.py --dimension 3 --dataset ugc_d_ext_1_500_11_21_11 --models hmonet meshonet pimeshonet --grids 11x21x11 21x41x21 41x81x41 --values 0.20 1.45 --seed 20261030 --run-tag v1

# tapered elliptic-bore block
python scripts/train/train_evaluate.py --dimension 3 --dataset teb_b_ext_1_500_11_21_11 --models hmonet meshonet pimeshonet --grids 11x21x11 21x41x21 41x81x41 --values 0.08 0.36 --seed 20261030 --run-tag v1

# bent annular duct
python scripts/train/train_evaluate.py --dimension 3 --dataset bad_a_ext_1_500_11_21_11 --models hmonet meshonet pimeshonet --grids 11x21x11 21x41x21 41x81x41 --values 10 105 --seed 20261030 --run-tag v1

# lobed annular duct
python scripts/train/train_evaluate.py --dimension 3 --dataset lad_a_ext_1_500_11_21_11 --models hmonet meshonet pimeshonet --grids 11x21x11 21x41x21 41x81x41 --values 0 0.20 --seed 20261030 --run-tag v1
```

服务器多 GPU 队列：将任一命令末尾追加 `--gpus 0 1 2 3`。同一数据集的 HMONet、MeshONet、PI-MeshONet 会作为三个独立的训练—评估任务并行分配至可用 GPU；全部结束后自动写出统一的多模型对比结果。
