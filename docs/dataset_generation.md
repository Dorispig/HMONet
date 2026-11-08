# 参数化数据集、边界族与曲线接口

项目只维护当前的“参数边界对象 + 边界采样点 + PDE 网格”格式。数据集名称遵循：

```text
<family>_<int|ext>_<version>_<size>_<grid shape>
```

例如 `air_t_ext_1_500_21_21` 和 `sch_r_ext_0_1000_11_31_11`。只改变样本数或网格规模时保留版本号；改变参数范围、几何公式或边界定向时递增 `version`，不得用旧名称覆盖新定义。

## 二维生成

```bash
python src/data/generation/make_dataset_parametric_2d.py --family circle_airfoil_t --split extrapolation --version 1 --size 500 --grid-shape 21x21 --parameter-range 0.055 0.6 --workers 4
```

主要参数：

- `--family`：边界族注册键；
- `--split`：`interpolation/int` 或 `extrapolation/ext`；
- `--version`：非负人工版本号；
- `--size`/`--count`：几何样本数；
- `--grid-shape NXxNY` 或 `--nx/--ny`：PDE 网格节点数；
- `--parameter-range MIN MAX`：单一边界族和划分的参数范围，可重复给出不连续区间；
- `--workers`：并行 PDE 真值生成进程数，`1` 为串行；
- `--output-dir`：显式输出目录；
- `--winslow-max-iter`、`--winslow-tolerance`：PDE 求解停止参数。

当前二维族覆盖翼型孔、偏心半环、半椭圆、圆角深 U 形凹槽、收缩通道、S/波纹通道、三角凸起和样条边界。准确注册键以生成器 `--help` 和源码注册表为准；已生成数据及参数范围见 [data/raw/2D/README.md](../data/raw/2D/README.md)。

## 三维生成

```bash
python src/data/generation/make_dataset_parametric_3d.py --family sch_r --split extrapolation --version 1 --size 500 --grid-shape 11x31x11 --seed 20261030
```

三维入口支持族、划分、版本、规模、网格尺寸、随机种子、PDE 迭代参数和输出根目录；当前不支持命令行 `--parameter-range` 或 `--workers`。正式边界以 `sch_r` 为主，新增参数面必须先检查六面方向、角点一致性、TFI 自交和极端参数合法性。已生成数据见 [data/raw/3D/README.md](../data/raw/3D/README.md)。

## 文件和安全规则

每个数据集至少包含 `dataset.npy`、`boundary_geometry.npy` 和元数据，两类数组的样本顺序必须一致。生成器不会静默覆盖同名目录。二维自定义单数据集会检查参数范围两端的极端网格；检查失败时仍按要求落盘并发出警告，此类数据只能用于诊断，不能直接作为论文监督真值。

生成后应核对：

- PDE 是否达到指定收敛标准；
- Jacobian 是否非正、单元是否翻转或相交；
- 参数曲线/曲面方向及共享角点是否一致；
- 数据集名称、元数据和真实参数范围是否一致。

不传 `--family` 时按源码 `SELECTED_DATASETS` 批量生成。论文复现应显式写出族、划分、版本、规模和网格，避免注册表变化影响任务。

## 参数曲线张量接口

`curve(t)` 保持 NumPy/Torch 输入后端，`curve.derivative(t)` 返回相对于全局 `[0,1]` 参数的导数。直线、圆、椭圆、NACA4、Bezier、B-spline、映射和分段曲线使用公式导数；拟合曲线使用分段线性斜率；通用函数使用显式导数或有界有限差分。

```python
import torch
from src.data.geometry.parametric_curve_2d import ParametricCurve, BoundaryCurveSet2D

curves = ParametricCurve.batch(curve_objects, device="cuda", dtype=torch.float32)
points = curves(t)                # t: [B,N], points: [B,N,D]
tangents = curves.derivative(t)   # 与 points 同形状

boundary = BoundaryCurveSet2D.batch(boundary_objects, "cuda", torch.float32)
projections = boundary.project_torch(xi, eta)  # [B,N,4,2]
```

驻留对象应每个 batch 构造一次并在多个前向/损失调用间复用；查询值和投影坐标不做记忆化缓存。NACA 的 `sqrt(x)` 参数化在前缘具有奇异切向量，端点采用有限张量约定，不能解释为该奇点处有限的解析导数。自定义 NumPy callable 若需要 GPU 可微计算，必须实现 `evaluate_torch`；修改几何系数、设备或 dtype 后应重建驻留对象。

