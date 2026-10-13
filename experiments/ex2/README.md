# ex2：HMONet 网络结构消融

冻结基线采用正式训练参数，Jacobian 权重固定为 `1e-4`。比较主干几何系数、自注意力和几何系数激活函数。默认随机种子为 `20260802、20260917、20261031`。

```powershell
python scripts/experiments/run_ablations_2d.py --experiment-set ex2 --ablations all --run-tag paper_v1
```
