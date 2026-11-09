# 物理采样策略消融

基线继承正式二维 HMONet 配置，即 200 个全局均匀点与 200 个曲率--边界距离自适应点。所有变体只改变每批 400 个物理中心的组成。

| 名称 | 内容 |
| --- | --- |
| `baseline` | 200 均匀 + 200 自适应（正式默认） |
| `e1_1` | 400 个均匀点 |
| `e1_2` | 300 均匀 + 100 自适应 |
| `e1_3` | 100 均匀 + 300 自适应 |
| `e1_4` | 400 个自适应点 |

## 命令

完整训练并评估：

```bash
python scripts/experiments/run.py ex2 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1
```

只训练指定策略：

```bash
python scripts/experiments/run.py ex2 e1_1 e1_4 --dataset air_t_ext_1_500_21_21 --seed 20261030 --tag v1 --train-only
```

只评估指定策略：

```bash
python scripts/experiments/run.py ex2 e1_1 e1_4 --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1 --evaluate-only
```

