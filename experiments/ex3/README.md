# ex3：Jacobian 与曲边目标插值组合消融

冻结基线使用双线性目标且关闭 Jacobian 损失。比较仅 Jacobian、仅曲边修正和二者同时开启；基线也作为第四个组合参与三随机种子汇总。

```powershell
python scripts/experiments/run_ablations_2d.py --experiment-set ex3 --ablations all --run-tag paper_v1
```

