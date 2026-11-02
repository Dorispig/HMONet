# ex6：训练数据集大小消融

本套实验运行800、500和250条训练数据，并与只读的1000条基线结果比较。所有数据规模统一使用
启用曲边真实值插值修正与Jacobian损失的HMONet基线，Jacobian损失权重
为 `1e-4`。800、500和250条数据通过原数据集索引的等间隔采样直接建立
内存子集，包含首尾样本并覆盖完整参数范围，不写入新数据文件。

配置修改前从 `ex3` 复制的旧基线不再与当前ex6基线一致，不会作为当前
实验的1000条基线复用。运行下列命令只会训练三个子集实验：

```powershell
python scripts/experiments/run_ablations_2d.py --experiment-set ex6 --ablations dataset_size --dataset air_t_ext_0_1000_41_20 --seeds 20260802 20260917 20261031 --run-tag paper_v1
```

若改用其他数据集或随机种子，应先建立对应的 `ex6` 基线：

```powershell
python scripts/experiments/run_baseline_2d.py --experiment-set ex6 --dataset <数据集名称> --seeds 20260802 20260917 20261031 --run-tag paper_v1 --evaluate
```



