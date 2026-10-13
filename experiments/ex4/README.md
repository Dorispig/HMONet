# ex4：精简 HMONet 的训练策略组合消融

本套实验采用约 17.8 万参数的 HMONet，比较普通双线性目标、Jacobian
损失、曲边目标插值以及二者同时启用。网络容量固定，其他训练和评估
参数继承 `baseline/experiment.yaml` 与 `baseline/evaluate.yaml`。

```powershell
python scripts/experiments/run_baseline_2d.py --experiment-set ex4 --dataset air_t_ext_0_1000_41_20 --seeds 20260802 20260917 20261031 --run-tag paper_v1 --evaluate

python scripts/experiments/run_ablations_2d.py --experiment-set ex4 --ablations jacobian_interpolation --run-tag paper_v1
```

第一条命令建立本套实验的规范基线；第二条命令运行四种组合。相同配置
和随机种子再次运行时会复用完整训练与评估，不会重复训练。
