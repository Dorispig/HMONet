# 硬边界约束消融

基线继承正式二维 HMONet 配置；所有变体只修改硬边界公式。默认数据集为 `air_t_ext_1_500_21_21`，默认评估参数为 `0.62 0.65`。

| 名称 | 内容 |
| --- | --- |
| `baseline` | CCAP 角点校正仿射投影（正式默认） |
| `e1_1` | RAP 有理仿射投影 |
| `e1_2` | Coons 基函数与网络残差 |

## 命令

完整训练并评估：

```bash
python scripts/experiments/run.py ex1 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1
```

只训练指定变体：

```bash
python scripts/experiments/run.py ex1 e1_1 e1_2 --dataset air_t_ext_1_500_21_21 --seed 20261030 --tag v1 --train-only
```

只评估指定变体：

```bash
python scripts/experiments/run.py ex1 e1_1 e1_2 --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1 --evaluate-only
```

`baseline all` 表示基线加全部变体；只写 `all` 表示仅全部变体，不重复运行基线。

