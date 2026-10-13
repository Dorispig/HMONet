# ex6：训练数据集大小消融

本套实验与 `ex3` 使用同一套HMONet基线参数，比较1000、800、500和
250条训练数据。800、500和250条数据通过原数据集索引的等间隔采样
直接建立内存子集，包含首尾样本并覆盖完整参数范围，不写入新数据文件。

`air_t_ext_0_1000_41_20` 的三个1000条基线结果已从 `ex3` 复制，运行
下列命令时会直接复用：

```powershell
python scripts/experiments/run_ablations_2d.py --experiment-set ex6 --ablations dataset_size --dataset air_t_ext_0_1000_41_20 --seeds 20260802 20260917 20261031 --run-tag paper_v1
```

若改用其他数据集或随机种子，应先建立对应的 `ex6` 基线：

```powershell
python scripts/experiments/run_baseline_2d.py --experiment-set ex6 --dataset <数据集名称> --seeds 20260802 20260917 20261031 --run-tag paper_v1 --evaluate
```
