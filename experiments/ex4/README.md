# PDE 求导区域与求导方式消融

基线继承正式二维 HMONet 配置。变体仅覆盖有限差分中心的合法域限制或导数后端。

| 名称 | 内容 |
| --- | --- |
| `baseline` | 中心差分；中心距逻辑边界至少一个差分步长 |
| `e1` | 中心覆盖整个开域，中心差分查询可越界 |
| `e2` | 自动微分替代有限差分 |
| `e3` | 边界一个步长内采用向内二阶单边差分 |

## 命令

完整训练并评估：

```bash
python scripts/experiments/run.py ex4 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1
```

只训练边界差分变体：

```bash
python scripts/experiments/run.py ex4 e1 e3 --dataset air_t_ext_1_500_21_21 --seed 20261030 --tag v1 --train-only
```

只评估自动微分变体：

```bash
python scripts/experiments/run.py ex4 e2 --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1 --evaluate-only
```
