# PDE求导区域与求导方式消融

基线直接继承当前正式二维配置（CCAP、归一化 PDE/Jacobian/SVD 损失及 200 个全局均匀点 + 200 个曲率—边界距离自适应点）。三个变体只覆盖求导区域或求导后端；旧监督训练与曲边插值字段已移除。

- `baseline`：中心限制在距逻辑边界至少一个差分步长处；
- `e1_fd_full_open_domain`：中心覆盖整个开域，中心差分查询允许越界；
- `e2_autograd`：使用自动微分；
- `e3_fd_one_sided_boundary`：边界一个步长内使用向内二阶单边差分。

```bash
python scripts/experiments/run.py ex4 baseline e1_fd_full_open_domain e2_autograd e3_fd_one_sided_boundary --dataset air_t_ext_1_500_21_21 --seed 20261030 --grids 101 201 401 --values 0.62 0.65 --tag v1
```

运行全部变体时将实验列表替换为`baseline all`。
