# 二维六类外推正式实验：临时命令清单

本文件用于逐个训练并评估六类二维外推数据集。每条命令均依次完成 HMONet、MeshONet、PI-MeshONet 的训练与评估；评估网格为 `101×101`、`201×201`、`401×401`。

在项目根目录 `E:\projects\HMONet` 运行。运行前请确认对应数据集已生成；数据集生成命令见 [data/2D/generation_commands.md](../../data/2D/generation_commands.md)。

```powershell
# circle_airfoil
python scripts/train/train_evaluate.py --dimension 2 --dataset air_t_ext_1_500_21_21 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.62 0.65 --seed 20261030 --run-tag v1

# elliptic_step_airfoil
python scripts/train/train_evaluate.py --dimension 2 --dataset esa_ext_1_500_21_21 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.72 0.75 --seed 20261030 --run-tag v1

# rounded_step_expansion
python scripts/train/train_evaluate.py --dimension 2 --dataset rse_ext_1_500_21_21 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.42 0.46 0.50 0.55 0.60 --seed 20261030 --run-tag v1

# rounded_u_notch
python scripts/train/train_evaluate.py --dimension 2 --dataset run_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 1.52 1.55 --seed 20261030 --run-tag v1

# smooth_constricted_channel
python scripts/train/train_evaluate.py --dimension 2 --dataset scc_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.42 0.45 0.47 --seed 20261030 --run-tag v1

# variable_s_wave_channel
python scripts/train/train_evaluate.py --dimension 2 --dataset vsc_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.85 0.90 0.95 --seed 20261030 --run-tag v1
```

服务器多 GPU 队列：将任一命令末尾追加 `--gpus 0 1 2 3`。同一数据集的 HMONet、MeshONet、PI-MeshONet 会作为三个独立的训练—评估任务并行分配至可用 GPU；全部结束后自动写出统一的多模型对比结果。
