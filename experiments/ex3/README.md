# ex3：Jacobian 与曲边目标插值组合消融

冻结基线使用双线性目标且关闭 Jacobian 损失，并仅作为只读参照。实验只运行仅 Jacobian、仅曲边修正和二者同时开启三种变体，不在本实验目录中重复运行基线。

```powershell
python scripts/experiments/run_ablations_2d.py --experiment-set ex3 --ablations all --run-tag paper_v1
```

