# ex5：HMONet 参数量消融

本套实验冻结 `ex3` 的训练和评估基线，只改变网络容量。五档模型在
`41x20` 输入网格下约为 56.2万、18.4万、8.5万、4.3万和1.69万参数。
程序会根据实际数据集网格重新计算参数量并将真实值写入汇总结果。
最小档约为同网格二维 MeshONet参数量的80%。最大档复用从 `ex3`
复制的完整基线训练结果。

```powershell
python scripts/experiments/run_ablations_2d.py --experiment-set ex5 --ablations model_capacity --dataset air_t_ext_0_1000_41_20 --seeds 20260802 20260917 20261031 --run-tag paper_v1
```

上述数据集和三个随机种子的完整基线已从 `ex3` 复制，因此最大模型会
直接复用。若改用其他数据集或随机种子，先用 `run_baseline_2d.py` 和
`--experiment-set ex5` 建立对应的规范基线。
