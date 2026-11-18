# 二维六类外推正式实验：临时命令清单

本文件用于逐个训练并评估六类二维外推数据集。六个训练数据集均为 `41×41`；每条命令均依次完成 HMONet、MeshONet、PI-MeshONet 的训练与评估，评估网格为 `101×101`、`201×201`、`401×401`。默认采用 HMONet 原始硬边界输出，MeshONet 与 PI-MeshONet 采用 IDW 精确边界后处理。

在项目根目录 `E:\projects\HMONet` 运行。运行前请确认对应数据集已生成；数据集生成命令见 [data/2D/generation_commands.md](../../data/2D/generation_commands.md)。

```powershell
# circle_airfoil_t (air_t)
python scripts/train/train_evaluate.py --dimension 2 --dataset air_t_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.62 0.65 --seed 20261030 --run-tag v1 --evaluate-args "--hmonet-boundary-projection none --meshonet-boundary-projection idw --pimeshonet-boundary-projection idw"

# eccentric_semiannulus (esa)
python scripts/train/train_evaluate.py --dimension 2 --dataset esa_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.72 0.75 --seed 20261030 --run-tag v1 --evaluate-args "--hmonet-boundary-projection none --meshonet-boundary-projection idw --pimeshonet-boundary-projection idw"

# rect_top_semiellipse (rse)
python scripts/train/train_evaluate.py --dimension 2 --dataset rse_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.42 0.46 0.50 0.55 0.60 --seed 20261030 --run-tag v1 --evaluate-args "--hmonet-boundary-projection none --meshonet-boundary-projection idw --pimeshonet-boundary-projection idw"

# rounded_u_notch_compact (run)
python scripts/train/train_evaluate.py --dimension 2 --dataset run_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 1.52 1.55 --seed 20261030 --run-tag v1 --evaluate-args "--hmonet-boundary-projection none --meshonet-boundary-projection idw --pimeshonet-boundary-projection idw"

# smooth_constricted_channel (scc)
python scripts/train/train_evaluate.py --dimension 2 --dataset scc_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.42 0.45 0.47 --seed 20261030 --run-tag v1 --evaluate-args "--hmonet-boundary-projection none --meshonet-boundary-projection idw --pimeshonet-boundary-projection idw"

# variable_width_s_channel (vsc)
python scripts/train/train_evaluate.py --dimension 2 --dataset vsc_ext_1_500_41_41 --models hmonet meshonet pimeshonet --grids 101 201 401 --values 0.85 0.90 0.95 --seed 20261030 --run-tag v1 --evaluate-args "--hmonet-boundary-projection none --meshonet-boundary-projection idw --pimeshonet-boundary-projection idw"
```

服务器多 GPU 队列：将任一命令末尾追加 `--gpus 0 1 2 3`。同一数据集的 HMONet、MeshONet、PI-MeshONet 会作为三个独立的训练—评估任务并行分配至可用 GPU；各任务保存自身、TFI、PDE 的评估产物，全部结束后联合阶段复用这些网格，仅生成统一的多模型质量汇总和对比图。
