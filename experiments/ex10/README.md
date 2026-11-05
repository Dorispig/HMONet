# ex10：采样策略与算子模型对比

基线采用 ex9/e9_3 的 HMONet、`current` 硬约束、尺度归一化 PDE、归一化 Jacobian 和最小奇异值损失；采样为 200 个全局均匀点、100 个曲率自适应点和 100 个边界内侧点，初始学习率为 `4e-4`。

六个实验依次比较：400 点全局均匀采样、400 点全自适应采样、200 全局加 200 曲率自适应采样、200 全局加 200 边界层采样、PI-MeshONet 和 MeshONet。默认数据集为 `air_t_ext_1_500_21_21` 与 `run_ext_1_500_41_41`，默认评估网格为 101、201、401。

注意：`run_ext_1_500_41_41` 当前有 51/500 个 PDE 样本包含局部非正角点行列式单元。程序允许按要求运行该数据集，但在将 MeshONet 的监督结果用于论文结论前，应先修复或重新生成这些训练真值。

## 当前实验名

- `uniform_400`：400个全局均匀随机点；
- `adaptive_400`：400个曲率-边界距离自适应采样点；
- `global200_adaptive200`：200个全局点+200个曲率自适应点；
- `global200_boundary200`：200个全局点+200个边界内侧点；
- `pimeshonet`：PI-MeshONet；
- `meshonet`：MeshONet。

`all` 运行上述六项但不包含基线；显式写 `baseline all` 才运行基线和六项实验。

```bash
python scripts/experiments/run.py ex10 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.52 0.55 --tag physics_v1
```

Linux 四卡动态运行：

```bash
python scripts/experiments/run.py ex10 baseline all --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.52 0.55 --tag physics_v1 --gpus 0 1 2 3
```

运行 `run_ext_1_500_41_41` 时把数据集替换为该名称，并使用 `--values 1.52 1.55`。评估默认不包含单元面积指标。
