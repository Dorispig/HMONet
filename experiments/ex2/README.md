# ex2：HMONet 网络结构消融

冻结基线采用正式训练参数，Jacobian 权重固定为 `1e-4`。当前保留主干几何系数与自注意力结构实验；Softplus 激活变体已从代码和实验中删除。默认随机种子为 `20260802、20260917、20261031`。

```powershell
python scripts/experiments/run.py ex2 all --tag paper_v1
python scripts/experiments/run.py ex2 baseline all --tag paper_v1
```

当前实验名为 `boundary_self_attention` 和 `trunk_geometry_coefficients`。历史结果若提到 Softplus，仅作为归档证据。
