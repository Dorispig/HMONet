# ex4：精简 HMONet 的训练策略组合消融

本套实验以 ex5 的 compact_178k 结构为基础，将四组隐藏层宽度统一为64；
在41x20输入下共有152522个可训练参数。普通双线性结果由只读基线提供；
实验只运行Jacobian损失、曲边目标插值以及二者同时启用。网络容量固定，其他训练和评估
参数继承 `baseline/experiment.yaml` 与 `baseline/evaluate.yaml`。

```powershell
python scripts/experiments/run_baseline_2d.py --experiment-set ex4 --dataset air_t_ext_0_1000_41_20 --seeds 20260802 20260917 20261031 --run-tag paper_v1 --evaluate

python scripts/experiments/run_ablations_2d.py --experiment-set ex4 --ablations jacobian_interpolation --run-tag paper_v1
```

第一条命令仅在用户明确需要时建立本套实验的规范基线；第二条命令只运行三种实验变体，不会自动运行基线。相同配置
和随机种子再次运行时会复用完整训练与评估，不会重复训练。
