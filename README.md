# HMONet

HMONet 是使用参数曲线（二维）和参数曲面（三维）的网格生成、训练与评估项目。代码文件统一使用分行和缩进格式。

## 目录

```text
HMONet/
├── configs/2D, configs/3D/       # 分维度训练与评估配置
├── data/raw/2D, data/raw/3D/     # 二维两文件数据集；三维沿用现有格式
├── scripts/                      # train/evaluate、TFI、PDE
├── src/data/                     # 数据集读取与分组后的数据处理模块
│   ├── dataset.py                # 二维/三维数据集读取
│   ├── geometry/                 # 参数曲线、曲面与边界构造
│   ├── generation/               # 数据集生成、说明、示例与合法性检查
│   ├── mesh/                     # topology.py、sampling.py、projection.py
│   └── backend/                  # Tensor求值、批量处理与加速内核
├── src/models/                   # HMONet、MeshONet
├── src/evaluation/               # 质量与局部图辅助函数
└── tests/                        # 流程与工具测试
```


### 数据模块调用路径

- 参数曲线及曲面：`src.data.geometry`。
- 数据集生成和检查：`src.data.generation`。
- 网格类与索引：`src.data.mesh.topology`；均匀采样与网格插值：`src.data.mesh.sampling`；边界后处理：`src.data.mesh.projection`。
- Tensor求值与加速：`src.data.backend`，实现说明见 [Tensor曲线说明](docs/README_tensor_curves.md)。
- 数据读取入口仍为 `src.data.dataset`。数据格式、训练和评估入口保持不变。

二维生成命令为 `python -m src.data.generation.make_dataset_parametric_2d`，三维生成命令为 `python -m src.data.generation.make_dataset_parametric_3d --name <数据集名称>`。
旧的扁平模块路径已迁移；自定义脚本应按上述路径更新导入。`parametric_grid.py` 合入 `mesh/sampling.py`，未使用的旧 `process.py`、自适应采样演示已移除。

## PI 自监督模型

支持二维、三维 PI-MeshONet 与 PI-HMONet。两者共享有限差分 PDE 残差，后者加入精确参数边界硬约束；四种模型共用收敛管理。
训练命令 `python scripts/train/train_2d.py --model pihmonet`；`--model all` 训练全部四种模型。
完整输入输出、损失公式、命令行和论文对照见 [PI 模型说明](scripts/train/pi_models.md)。

## 数据集

新生成的二维数据集只保存两个文件：`dataset.npy` 存边界采样点与 PDE 网格，`boundary_geometry.npy` 存参数边界描述。两个文件的相同索引严格对应同一个样本。二维边界由 bottom/right/top/left 四条参数曲线构成，二维读取器专用于此两文件格式。三维生成器仍使用原有文件布局，边界由六个参数曲面构成；三维读取器只加载边界点、网格，以及 HMONet 需要的参数曲面，不加载内部采样点文件。

二维数据集生成代码统一放在 `src/data/generation/make_dataset_parametric_2d.py`。当前 `SELECTED_DATASETS` 包含9类数据集的18个内插/外推选项，每套默认1000条；运行前请只保留需要生成的选项。下表列出其中与旧项目对应的三类：

| 数据族 | 变化参数 | 内插训练区间 | 外推训练区间 | 默认网格 |
| --- | --- | --- | --- | --- |
| `rect_top_semiellipse` / rse | `semi_minor_axis` | `[0,0.2]`、`[0.3,0.5]` | `[0,0.4]` | 20×20 |
| `eccentric_semiannulus` / esa | `inner_center_x` | `[0.3,0.45]`、`[0.55,0.7]` | `[0.3,0.7]` | 20×20 |
| `symmetric_convex_tri` / tri | `protrusion_distance` | `[0.05,0.2]`、`[0.3,0.5]` | `[0.05,0.4]` | 21×20 |

以上三类内插与旧项目一致，每个区间均匀取500个参数，外推训练区间均匀取1000个参数。训练区间、固定几何和评估参数均对齐旧项目。

```powershell
python -m src.data.generation.make_dataset_parametric_2d --size 1000 --timing-samples 3
```

在文件顶部修改 `SELECTED_DATASETS` 选择需要生成的任务；已存在的同名目录不会被覆盖。`DATASET_GENERATORS` 登记参数边界构造函数，`DATASET_FAMILIES` 记录名称、参数、区间、固定几何和默认网格，`SPLIT_ALIASES` 支持 `int/interpolation`、`ext/extrapolation`。

`DEFAULT_GENERATION_OPTIONS` 的 `nx/ny=None` 表示使用各数据族默认规模；命令行指定 `--grid-shape` 或 `--nx/--ny` 时优先采用用户输入。tri 要求 nx 为奇数，确保三角尖点被采样。输出名称按实际规模构造，例如 `rse_0_1000_20_20`、`esa_1_1000_20_20`、`tri_0_1000_21_20`。

`circle_airfoil_t` 已列入默认选项；如果不需要重新生成，应在选择列表中注释其两项。它的内插区间仍为 `[0.05,0.25]`、`[0.35,0.60]`，外推训练区间为 `[0.05,0.60]`，原有生成接口保持可用。

批量入口为 `generate_selected_datasets(options, output_dir=None)`，依次调用 `generate_dataset(family, split, ...)`。单套入口按旧项目的顺序规范化划分名称、读取配置、生成参数值、构造边界、均匀采样边界、求解 Winslow 网格和保存结果。几何构造使用新项目的参数曲线，并保留此前要求的等间隔参数采样和两文件保存方式。新增已实现的二维边界时，在顶部注册构造函数、参数名和区间即可复用该流程，不需要把 `t` 或 `circle_airfoil` 写死在批量入口中。

`generate_circle_airfoil_t_dataset()` 仍提供单独调用与自定义区间的接口。区间采样 `sample_parameter_ranges()`、PDE 求解及边界检查 `_solve_pde_grid()`、两文件生成流程 `_generate_pde_dataset()` 为共用函数。生成阶段不生成随机内部采样点，也不对网格额外插值。

`to_computational_boundary(boundary)` 接收任意符合四边逻辑结构的 `BoundaryCurveSet2D`，统一 TFI 和 HMONet 所需的参数方向，不修改输入对象。它使用通用的 `metadata["parameter_order"]` 约定：`computational` 表示上下曲线随 xi 递增、左右曲线随 eta 递增；`perimeter` 表示四条曲线已按 bottom/right/top/left 连成一圈，转换时反转上、左曲线。没有方向标记时通常按 `computational` 处理，旧版圆加翼型描述仍兼容其原先的一圈方向。新生成的圆加翼型边界会明确标记 `perimeter`。

转换只统一方向并检查端点连接，不根据物理坐标猜测边界归属，也不改变曲线形状或采样密度。自定义的一圈方向边界可以设置元数据，或直接传 `source_order="perimeter"`。例如：

```python
from src.data.geometry.boundary_generator_2d import generate_boundary
from src.data.generation.make_dataset_parametric_2d import to_computational_boundary

boundary = to_computational_boundary(generate_boundary("rect_top_semiellipse"))
```

命令行支持原项目风格的 `--size`、`--grid-shape 21x21`、`--winslow-max-iter`、`--winslow-tolerance`、`--timing-samples` 和 `--output-dir`。兼容现有的 `--count`、`--pde-max-iter`、`--pde-tolerance` 别名，也可以使用 `--nx/--ny` 单独覆盖网格点数；`--grid-shape` 与 `--nx/--ny` 不同时指定。选项覆盖文件顶部对应的公共配置。

生成选择和命名统一由代码控制，`--name`、`--dataset-type` 不再使用。`m`、`p`、`c`、外圆半径、圆心和 `t` 区间均不提供命令行选项。旧三维脚本的随机参数采样不用于这里：按已确认要求，几何参数与边界曲线参数均采用等间隔采样，不需要生成阶段的 `--seed`。

`generate_circle_airfoil_t_dataset()` 未指定 `t_ranges` 时，从 `DATASET_FAMILIES` 读取对应 `split` 的区间，默认使用内插区间。每个区间内等间隔取值并包含两个端点，数量沿用本项目按区间长度比例分配的规则：默认 1000 条内插数据分别为 444、556 条；外推在连续区间内取 1000 条。原项目内插是每段 500 条，这里的区间范围相同、数量分配规则不同。重叠或相接的区间先合并，不重复生成相同的 `t`。

`t` 是翼型厚度参数，曲线内部的采样参数仍为 `[0,1]`。固定几何参数默认为 `m=0.02, p=0.4, c=1.0, radius=1.0, center=(0.0, 0.0)`。批量生成时，区间与固定参数均修改顶部 `DATASET_FAMILIES`，固定参数位于 `fixed_parameters`。直接调用 `generate_circle_airfoil_t_dataset()` 时，也可传入 `t_ranges` 和固定参数；显式区间未指定 `split` 时，记录为 `custom`。

`nx`、`ny` 是包含首尾端点的点数：外圆和翼型各取 `nx` 点，两条连接线各取 `ny` 点，均在曲线参数域内均匀采样。数据集边界数组按计算域边界顺序保存，公共角点不重复占用同一个逻辑节点，共有 `2*(nx+ny)-4` 个节点。上下曲线使用递增 `xi` 方向，左右曲线使用递增 `eta` 方向；绕边界遍历时将上、左曲线反向。`nx` 为奇数时包含翼型参数 `0.5` 的左端点；偶数时保持均匀采样和用户指定点数，不额外插点。

网格真值使用 Winslow PDE 迭代结果，TFI 仅作为初值。`--pde-max-iter` 默认 1000，`--pde-tolerance` 默认 `1e-6`，采用求解器的相邻迭代网格相对变化量判断收敛。未收敛或出现非有限值时报告对应样本并终止，不将 TFI 冒充 PDE 真值。输出前检查 PDE 网格边界与采样点一致。`--timing-samples` 控制前几条单样本耗时输出，后续仍每 10% 输出进度。

默认输出为项目下的 `data/raw/2D/<name>/`，不受命令启动目录影响，可通过 `--output-dir` 覆盖根目录。每个数据集一个文件夹，同名目录已存在时拒绝覆盖；全部样本成功后才发布最终目录。统一入口使用 `output_dir`，单独的翼型生成函数与显式参数生成函数仍使用 `output_root`。

每次生成成功后，自动更新 [data/raw/2D/README.md](data/raw/2D/README.md)，说明本目录实际存在的每个数据集：边界形状、样本数量、网格大小、固定参数、变化参数范围、区间数量分配和 PDE 配置等。每个文件的第一条记录附带 `metadata`，不占用额外样本索引。参数文件的 `metadata.fixed_parameters` 记录全部固定几何参数，`metadata.parameter_ranges` 记录变化区间，还记录每条数据的 PDE 迭代次数、收敛状态、耗时与边界误差。数值文件只附带网格尺寸、数量、文件配对标识等结构信息，MeshONet 不需要加载几何描述。

| 文件 | 内容 |
| --- | --- |
| `dataset.npy` | `(N,)` 字典数组；每条记录的 `boundary_points` 为 `(2*(nx+ny)-4, 2)` 边界点，`mesh_points` 为 `(ny*nx, 2)` PDE 网格点，均为 `float32`，字段名及展开方式与原项目一致 |
| `boundary_geometry.npy` | `(N,)` 字典数组；每条记录的 `boundary` 保存四条参数曲线的可还原描述，`parameters` 保存该样本的变化参数，如 `{"t": 0.12}` |

两个文件的每条记录还保存 `sample_id` 和 `dataset_id`。同一次生成共享一个唯一的 `dataset_id`，`sample_id` 等于从零开始的数组索引。HMONet 读取时检查文件配对、长度、顺序和网格大小；混用不同数据集的文件或打乱样本顺序会报错。读取示例：

```python
from src.data.dataset import ParametricDataset2D

path = "data/raw/2D/air_t_int_0_1000_21_20"
hmonet_data = ParametricDataset2D(path, load_boundary=True)
meshonet_data = ParametricDataset2D(path, load_boundary=False)

sample = hmonet_data[0]
boundary = sample["boundary"]          # BoundaryCurveSet2D instance
points = sample["boundary_points"]
mesh = sample["mesh_points"]           # Stored shape: (ny*nx, 2)
```

`load_boundary=False` 只打开 `dataset.npy`，不读取参数文件，也不实例化曲线类；`load_boundary=True` 则读取两个文件并还原边界类。默认值为 `True`，二维与三维使用相同开关。路径也可以直接指定到 `dataset.npy`。需要原始字典时，使用 `np.load(path, allow_pickle=True)`，第 `i` 条记录直接取数组的 `[i]`。

`dataset[i]` 只返回模型需要的字段：MeshONet 为 `boundary_points` 和 `mesh_points`，HMONet 再加 `boundary`。坐标数组保持文件中的键名与形状。文件中的 `sample_id`、`dataset_id` 和 `metadata` 用于读取校验，不放进训练样本或长期保存在类中；`parameters` 也不进入训练样本。评估需要参数标签时，单独调用 `read_sample_parameters(dataset.root, index)`；数据集说明需要生成配置时，调用 `read_dataset_metadata(dataset.root)`。

读取类的实例变量只有：`root`（数据集目录）、`load_boundary`（是否读取参数边界）、`grid_shape`（计算域网格点数）、`boundary_points`（所有边界采样点）、`mesh_points`（所有网格真值），以及仅在 `load_boundary=True` 时持有的 `boundary`（参数边界类列表）。二维、三维采用相同的样本输出规则，训练中的批量索引直接使用这些数组。

Python 生成函数也可直接调用：

```python
from src.data.generation.make_dataset_parametric_2d import generate_circle_airfoil_t_dataset

output = generate_circle_airfoil_t_dataset(
    name="air_t_int_0_1000_21_20",
    split="interpolation",
    nx=21,
    ny=21,
    count=1000,
    m=0.02,
    p=0.4,
    c=1.0,
)
```

单套自动命名入口示例为 `generate_dataset("circle_airfoil_t", "int", size=1000, nx=21, ny=20)`，支持完整或简写的 split 名称。原先接收边界类别和显式参数字典序列的接口改名为 `generate_explicit_dataset(name, boundary_type, parameters, ...)`，与其他入口共用 PDE 迭代和两文件保存流程。完整评估需要参数边界来计算 TFI、精确边界误差，因此读取两个文件；MeshONet 训练只读取数值文件。

## 训练

二维和三维分别训练。`--model both` 依次训练 HMONet 和 MeshONet，`--dataset-mode both` 依次使用 interpolation 与 extrapolation 数据集；`--batch-size` 覆盖配置中的 batch。

```powershell
python scripts/train/train_2d.py --config configs/2D/train.yaml --model both --dataset-mode both --epochs 100
python scripts/train/train_3d.py --config configs/3D/train.yaml --model both --dataset-mode both --epochs 100
```

每轮随机打乱样本。MeshONet 使用批内共享的均匀随机计算域点；HMONet 可通过 `hmonet_sampling.mode` 选择 `uniform`（批内共享的均匀随机点）或 `adaptive`（按每个样本的物理边界曲率和到计算域边界的距离加权采样）。训练损失按成功更新的 batch 的实际样本数加权平均。MeshONet 只读取边界采样点和 PDE 网格；HMONet 另外读取与数值样本顺序一致的参数边界对象。

二维 HMONet 的随机点和可选固定内部网格点主损失均为 `2 * mean((weight * (prediction - target)) ** 2)`，即每点加权平方距离的平均值。随机点权重为 `1 + 16 * ((xi - 0.5)^4 + (eta - 0.5)^4)`，固定点权重系数为 32；固定点默认关闭，可用 `--fixed-grid-count` 开启。MeshONet 保持 `alpha * interior_mse + beta * boundary_mse`。三维 HMONet 保留原有的加权坐标损失尺度，不套用二维乘 2 的系数。

固定网格 MSE 与训练损失分开：对全部样本的固定内部节点计算 `2 * mean((prediction - target) ** 2)`，排除边界节点、不加空间权重；二维、三维均乘 2。日志、图像和 MSE 收敛判定使用同一结果；最佳模型默认按逐轮训练损失的全局最小值选择，摘要记录 `dataset_mse_scale: 2.0`。默认 `dataset_mse_interval: 1`，每轮计算，可用 `--dataset-mse-interval` 覆盖。`hmonet_training` 和 `meshonet_training` 中的平台收敛开关分别生效，命令行 `--check-plateau-stability` / `--no-check-plateau-stability` 等可覆盖。学习率最后一次衰减允许低于停止阈值。非有限损失或梯度根据 `skip_nonfinite_loss` 跳过或报错；梯度裁剪可设置为 YAML 的 `null` 关闭。

二维 `configs/2D/train.yaml` 的随机种子为 `20260802`，与旧项目 air_t 两种数据集、两个模型的历史训练记录一致。

HMONet 不再使用 `boundary_curve_interpolation` 配置。传入参数边界时直接计算曲线或曲面上的投影点，缺少参数边界时使用二维分段线性插值或三维双线性插值。旧配置中的该字段会被兼容移除并提示；`hmonet_mapping_mode: current / coons_residual` 仍用于选择内部映射。

`current` 映射在二维和三维中采用一致的有理硬边界结构：二维使用四条边权重，三维使用六个参数面权重，中心系数均为 `1-sum(w_i)`，不执行凸组合归一化、法向 smoothstep 或切向有理抬升，并在参数边界上精确覆盖。HMONet 可选由边界分支输出全局几何系数，并由坐标主干 MLP 输出位置相关系数，两者逐位相乘。几何系数可选择 Softplus 或 Sigmoid 激活。边界输入还可在原分支网络前启用四边/六面的联合自注意力编码。正式配置默认开启主干几何系数、Softplus 和自注意力，三项均可通过配置或命令行关闭以进行消融实验。

参数边界在 `src/data/backend/parametric_torch.py` 中批量求值。直线、圆弧、椭圆弧、NACA 翼型、映射曲线和分段曲线直接在查询张量所在设备计算；三维双线性面、圆柱面、曲线拉伸面、端盖和分段面复用该路径。同类几何跨样本和边界侧面合并计算，每次前向仅临时上传几何参数，不保存参数或查询点缓存。训练、固定网格 MSE 和评估均自动使用此路径，模型权重结构和数据集格式不变。

自定义曲线或曲面可以覆盖 `evaluate_torch(t)` 或 `evaluate_torch(u, v)`，返回同设备的张量以保留参数坐标梯度。只有 NumPy 实现时，按组传输查询和结果，继续使用原曲线公式；该兼容路径会分离查询坐标梯度。当前 B 样条和拟合曲线也使用 NumPy 兼容路径，不将 B 样条替换为 Bezier 曲线。

训练日志目录为 `outputs/training/<2D或3D>/<dataset>/hmonet/<current或coons_residual>/<YYYYMMDD_HHMM>/`；MeshONet 为同级的 `meshonet/<YYYYMMDD_HHMM>/`。选择 `both` 时，每个数据集、每个模型分别训练和保存。每个运行目录包含：

```text
training_history.csv     # 旧版字段：二维 epoch,training_loss,dataset_mse；三维另有损失分量
mse_history.csv          # epoch,dataset_mse
training_loss.png        # 对数纵轴
dataset_mse.png          # 对数纵轴
training_summary.json    # 旧版摘要字段、训练耗时、最佳/最终 MSE、收敛状态和路径
training_summary.csv
config.json              # 应用命令行覆盖后的实际配置
config.yaml              # 三维训练目录保存；二维 YAML 保存在 checkpoint 目录
tensorboard/             # 启用时生成；使用旧版指标标签
```

模型单独保存到以下位置，日期和时分位于 HMONet 投影方式目录下：

```text
outputs/checkpoints/<2D或3D>/<dataset>/
  HMONet/<current或coons_residual>/<YYYYMMDD_HHMM>/
  MeshONet/<YYYYMMDD_HHMM>/
```

每个模型目录保存 `best.pt`、`best_model.pth`、`final.pt`、`final_model.pth`、`model_parameters.json`、`checkpoint_metadata.json`、实际配置和训练摘要。`.pt` 包含模型权重、配置和轮次；`.pth` 为纯 `state_dict`。训练函数继续分别放在两个维度文件的 `train_hmonet()` 和 `train_meshonet()` 中，`training_common.py` 提供数值工具，`training_state.py` 负责指标、收敛和输出，不包含训练循环。

每个模型训练完成后，终端打印实际/设定迭代次数、训练时间（秒及时分秒）、最佳模型轮次和选择指标、最佳模型的训练损失及固定网格 MSE、全程最低训练损失及轮次、最终损失/MSE、停止原因、学习率和保存路径。二维和三维全部所选模型/数据集训练结束后，再集中列出所有运行的摘要及总耗时，`both` 模式逐项区分。相同指标写入训练目录和 checkpoint 目录的 `training_summary.json`、`training_summary.csv`；启用 TensorBoard 时也写入两份日志的 `run/summary`。`best_training_loss` 是选中的最佳模型所在轮次的训练损失；`minimum_training_loss` 和 `minimum_training_loss_epoch` 是全程最低训练损失及轮次。按 `dataset_mse` 选模型时，这两个轮次可能不同。单模型训练时间沿用原统计口径，不包含结束后的补充测量、绘图和保存；最后的总耗时包含整个调度过程。

同一次训练的日志、checkpoint 和两份 TensorBoard 日志共用一个运行标识，例如 `20260915_1954`；同一分钟同一数据集、同一模型再次启动时使用 `_02`、`_03` 等后缀，避免覆盖或混合事件。三处输出的运行目录都会检查重名，独立 TensorBoard 目录已存在时也不会复用。运行标识在创建目录时一次性确定，训练跨过分钟或午夜也不改变目录。摘要中的 `run_id`、`training_output_dir`、`checkpoint_dir`、`tensorboard_log_dir` 和 `tensorboard_mirror_log_dir` 记录实际位置。

二维、三维 `train.yaml` 均启用 `training.tensorboard_enabled: true`。训练目录内的 `tensorboard/` 保留，同时在 `outputs/tensorboard/` 下按相同的维度、数据集、模型、投影方式和日期时分层级额外输出一份。独立目录的运行文件夹直接存放 event 文件：

```text
outputs/training/<维度>/<数据集>/
  hmonet/<current或coons_residual>/<日期_时分>/tensorboard/events.out.tfevents.*
  meshonet/<日期_时分>/tensorboard/events.out.tfevents.*

outputs/tensorboard/<维度>/<数据集>/
  hmonet/<current或coons_residual>/<日期_时分>/events.out.tfevents.*
  meshonet/<日期_时分>/events.out.tfevents.*
```

两份日志在训练期间同步写入相同的标量、步数、时间和配置/最终摘要文本；训练结束会刷新并关闭两份日志。记录包括训练损失、固定网格 MSE、学习率以及启用的几何损失指标。`--no-tensorboard-enabled` 同时关闭两份输出。训练记录文件按旧项目保留：二维使用配置 JSON，三维使用配置 JSON/YAML，均输出两份历史 CSV、两张曲线图和 JSON/CSV 摘要；不再额外生成重复的 `training_log.csv`、`loss_history.csv`。checkpoint 继续采用已确定的 best/final 四文件和配置、元数据结构。

在 MGNet 环境中可只加载独立目录查看全部运行：

```powershell
tensorboard --logdir outputs/tensorboard --port 6006
```

浏览器访问 `http://localhost:6006`，可按数据集、模型、投影方式和日期时分选择运行记录。

## 性能对比

在 MGNet 环境和项目根目录下运行：

```powershell
python scripts/benchmark/benchmark_hmonet.py --legacy-root E:\projects\HMONet --dimension both --mapping current
```

可用 `--mapping coons_residual` 比较另一种映射，或用 `--dataset` 指定另一套二维数据。默认 batch=64、每个样本 400 个随机点、单项计时 20 次；完整训练轮次先预热一轮，再测三轮。`--baseline-root` 可选，指向保留修改前模型和参数几何文件的项目备份，用于加入“优化前新项目”对照。

报告保存在 `outputs/benchmarks/hmonet_<日期时间>/`：`results.json` 记录环境、参数和每次计时，`timings.csv` 保存统计表，`comparison.md` 汇总比较。采用 CPU 墙钟并同步 CUDA，包含前向内部的参数打包与传输，不包含文件读取、绘图、checkpoint 保存或网络输出后的边界后处理。整轮计时使用相同训练驱动、相同初始权重和相同输入比较模型；二维读取实际 PDE 数据集，三维使用 64 个可解析 box 网格作为计时样本，不代表三维生产数据集的收敛效果。预测计时使用 batch=1 输出完整网格，同时另测 batch=64 的吞吐量。

## 评估

二维、三维评估分别读取各自的 `evaluate.yaml`。默认比较 TFI、PDE、MeshONet、HMONet；`--methods` 决定实际评估的方法和唯一的对比组，不再生成固定组合或占位模型结果。缺失 checkpoint 会明确报告并跳过；使用 `--no-skip-missing-checkpoints` 可要求全部指定模型必须存在。

```powershell
python scripts/evaluate/evaluate_2d.py --config configs/2D/evaluate.yaml
python scripts/evaluate/evaluate_2d.py --case air_t_int_0_1000_21_20 --methods tfi,pde --values 0.30 --grid-shapes 21x20
python scripts/evaluate/evaluate_2d.py --case air_t_int_0_1000_21_20 --values 0.30 --grid-shapes 51x50 --center-on worst --metric intersecting_cell
python scripts/evaluate/evaluate_3d.py --config configs/3D/evaluate.yaml --methods tfi,pde
```

`--dataset data/raw/2D/<dataset> --parameter-index 0` 仍可检查保存的数据集样本。正式内插/外推测试使用 YAML case 的留出参数，无需预先生成测试数据集。网络边界输入保持 checkpoint 的训练尺寸，评估查询网格独立设置。HMONet 输入恢复后的参数边界类。

输出根目录为 `outputs/results/<2D或3D>/<dataset>/<grid>/<参数值>/`。网络目录区分 `raw`（未经边界后处理）和 `direct`/`inverse_distance`/`none`；HMONet 还区分 `current`/`coons_residual`。原始及后处理边界误差分别保存到方法目录和参数目录的 JSON/CSV 中；误差为对应逻辑边界采样点到精确参数边界点的欧氏距离。

二维、三维 HMONet 默认 `none`，内部硬约束后直接使用网络输出，外部后处理耗时为 0；MeshONet 默认 `direct`。默认比较组为 `comparisons/current/hmonet_none__meshonet_direct/01_<所选方法>/`。原始边界误差继续计算和保存。网络计时默认预热 3 次、正式测量 10 次，去掉一次最大值和一次最小值后取剩余 8 次的平均。可用 `--inference-warmup-runs 3 --inference-timing-repeats 10` 修改次数；只有 1～2 次测量时使用普通平均。JSON 保留全部测量值、汇总方式和采用的样本数，CSV 中也注明汇总方式。

评估质量计算由 `evaluation.compute_grid_quality`、`evaluation.quality_metrics` 和 `evaluation.quality_switches` 控制，二维、三维配置文件均列出了各指标开关。`quality_metrics: both`（或 `all`）表示选择全部已实现指标，`none` 表示不选择，也可以设置列表，如 `[min_angle, scaled_jacobian]`。只有在选择列表内、且对应开关为 `true` 的指标才计算、保存和绘制；共享的几何中间量按所选指标的定义计算，不为未选指标额外运行检测。三维同名指标同时控制体网格及切片；`cell_area`、`scaled_jacobian` 等四边形指标用于中间切片，`cell_volume`、`minimum_scaled_jacobian` 等体网格指标用于六面体。

```powershell
# Only mesh figures/data; retain generation timing and boundary-error records.
python scripts/evaluate/evaluate_2d.py --case air_t_int_0_1000_21_20 --no-compute-grid-quality
# Compute only these two metrics.
python scripts/evaluate/evaluate_2d.py --case air_t_int_0_1000_21_20 --quality-metrics min_angle,scaled_jacobian
# All metrics except inter-cell intersection detection.
python scripts/evaluate/evaluate_2d.py --case air_t_int_0_1000_21_20 --quality-metrics both --no-quality-intersecting-cell
# Volume Jacobian and middle-section Jacobian.
python scripts/evaluate/evaluate_3d.py --case cah_t_int_0_1000_11_11_11 --quality-metrics minimum_scaled_jacobian,scaled_jacobian
```

`--quality-<指标名>` / `--no-quality-<指标名>` 覆盖对应布尔开关，命令行的指标名连接符使用 `-`，如 `--no-quality-min-angle`；`--quality-metrics` 列表中的指标名保持下划线。关闭总开关时，不计算二维、三维体网格或中间切片的质量，只保留网格图和数值记录；模型边界误差与 PDE 对照误差仍记录数值，不画误差颜色图。局部图同样遵守选择：可按坐标范围画网格局部图，关闭的指标不能用于质量图或 `worst` 搜索，此类任务会显示跳过原因。质量选择变化会重建分析并清理旧质量文件，网格生成缓存仍可复用；部分指标下的非法单元统计仅代表已检查的指标，不代表完整有效性检查。

HMONet 在每次前向中直接求二维四个/三维六个投影点：提供参数边界时使用精确曲线/曲面求值，否则插值采样边界。评估外部投影预计算已撤回，相应 YAML/命令行开关和 `projection_points` 输入已移除；现有模型权重无需重新训练。

Evaluation passes original boundary objects directly to HMONet. Coefficient preparation, Cartesian query deduplication and CUDA Graph capture have been removed. Batched evaluation remains available through `--inference-batch-size N`.

精确曲线求值默认 `curve_kernel_backend: auto`，可用 `--curve-kernel-backend auto|torch` 覆盖。auto 在可用的 CUDA 环境使用融合公式计算直线、圆/椭圆、NACA 翼型，以及兼容的两段 NACA 参数曲线；分段翼型只计算当前参数所属分段。二维边界和三维拉伸曲面/端盖中的这些曲线共用该实现，current 和 coons_residual 均受益。只缓存编译内核，不缓存曲线值；不同边界参数仍在每次前向传入。CPU、不支持的类型/精度或需要坐标梯度时继续使用原 Torch 公式，保留一阶和高阶求导。函数及检查点接口不变，已有权重无需重训；融合计算可能产生浮点舍入差异。

CUDA curve kernels may initialize or compile lazily during their first ordinary forward. Warmup count follows the configuration exactly, including zero; there is no separate setup stage or setup time column. Use `--curve-kernel-backend torch` to disable fused formulas. Kernel fallback reasons remain recorded in `curve_kernel_fallbacks`.

`inference_batch_size` 默认 1；大于 1 时合并同模型、同尺寸及同推理设置的参数案例，结果仍写入各自参数文件夹。CUDA Graph 默认关闭：开启后，在同一次评估进程中可复用兼容模型/查询/几何结构的执行图，更新边界输入与几何参数后重新执行所有投影运算；不缓存投影坐标。捕获成本仅在首次/结构变化时支付，最多保留 4 个执行图，进程退出后释放。不支持的情况自动回退，并在计时 JSON 中记录原因。

`network_inference_seconds` measures the complete model forward, including internal projection. `total_model_seconds` adds boundary postprocessing. Batched timing records `inference_batch_size`, `batch_inference_seconds` and `inference_time_basis`. Paper tables retain the forward-only Grid generation time column. Model loading, input sampling, output transfer, quality computation and plots are outside that column.

例如，同一模型的多个参数批量评估（PowerShell 单行命令）：

```powershell
python scripts/evaluate/evaluate_2d.py --case air_t_int_0_1000_21_20 --values 0.2,0.25,0.3 --grid-shapes 51x50 --inference-batch-size 3
```

The same direct inference path is used in 2D and 3D. `scripts/benchmark/benchmark_evaluation.py` compares ordinary model forwards and batch throughput; `scripts/benchmark/benchmark_curve_kernels.py` compares fused and Torch formulas. No benchmark path prepares geometry adapters or captures CUDA Graphs.

每个方法输出网格、全部对应质量数据/图和非法单元统计；对比组输出统一色标的横向比较、逐点/逐单元 PDE 误差（选择 PDE 时）与论文时间表。三维额外保存 `boundary_faces/x_mid`、`y_mid`、`z_mid` 的网格、内蕴质量和真实检测的自相交统计。整体和局部切片统一采用固定物理坐标平面：x_mid 为 YZ，y_mid 为 XZ，z_mid 为 XY，避免因自动相机改变视角；局部图也支持用户选择 PCA 投影。

默认只输出 PNG 质量着色图，不输出直方分布图。`--no-plots` 只保存数值文件。生成与分析分别按参数、checkpoint 内容及代码版本复用；`--force` 强制两者重做，`--force-regenerate`/`--force-reanalyze` 可单独控制，`--analysis-only` 只读取有效生成缓存。网格生成计时对网络只统计前向过程，另列边界后处理时间，预热次数和重复次数可配置。

单独重新生成局部图，不需要重新预测：

```powershell
python scripts/evaluate/plot_result_zoom.py --parameter-dir "outputs/results/2D/air_t_int_0_1000_21_20/21x20/0.3" --plot quality --metric intersecting_cell --xlim=-0.55,0.55 --ylim=-0.25,0.25
python scripts/evaluate/plot_result_zoom_3d.py --batch-root "outputs/results/3D/<dataset>/<grid>/<参数值>" --mode slice --slice z_mid --metric intersecting_cell --center-on worst
```

完整目录树、参数覆盖方式、验证范围与新旧项目差异见 [评估体系与输出说明](docs/evaluation_system.md)。三维 box 与翼型孔圆柱已修正参数方向和 TFI；没有实现精确工厂的旧管道/stadium 模板已禁用，不会再伪装成 box。旧三维生成器曾将 TFI 写入 `pde_mesh.npy`；现在保存实际 PDE 迭代结果，旧数据不会被自动改写，需要重新生成后再作为 PDE 真值使用。

## 测试

```powershell
python -m black --check src scripts tests
python -m unittest discover -s tests -v
```

运行模型和测试需要按 `requirements.txt` 安装 PyTorch、NumPy、SciPy、PyYAML、Matplotlib。


## Tensor curve API

See [batched curve values and analytic derivatives](docs/README_tensor_curves.md) for the resident four-edge projection API, endpoint conventions and training integration.

## Direct evaluation without setup

Evaluation uses original boundary objects with no coefficient-preparation adapter, Cartesian query preparation, or CUDA Graph capture. Setup options and timing columns have been removed. Warmup counts are honored exactly; lazy kernel initialization can occur on first use. Training retains its resident batch coefficients.


### RSE, ESA and TRI datasets

The 2D generator also supports `rect_top_semiellipse` (rse),
`eccentric_semiannulus` (esa), and `symmetric_convex_tri` (tri).
Their geometry defaults, training intervals, dataset names and held-out evaluation
values follow the original project. Each split defaults to 1000 samples;
interpolation uses 500 evenly spaced values in each interval.
RSE/ESA default to 20x20 nodes, TRI to 21x20 (odd nx preserves the apex).
All nine families are listed in `src/data/generation/make_dataset_parametric_2d.py`.
Comment out unneeded entries before generation; existing datasets are never overwritten.

Run `python -m src.data.generation.make_dataset_parametric_2d` in MGNet to generate the
selected jobs. `--size`, `--grid-shape`, PDE limits and timing options remain
available. Existing output folders are rejected rather than overwritten.
Each folder contains `dataset.npy` and `boundary_geometry.npy`, paired by sample
index and dataset identity. `data/raw/2D/README.md` lists actual completed datasets.
The renamed boundary constructors match the original Python names:
`GenerateRectangleWithTopSemiEllipseBoundary`, `EccentricAnnularSectorBoundary`,
and `VerticallyAsymmetricConvexTriBoundary`; they return parameter-curve objects,
so sampling sizes are supplied to the dataset generator, not these constructors.

## PI-MeshONet：二维与三维自监督训练

使用独立的论文版网络类 `PIMeshONet2D/3D`：坐标分支和共享主干采用门控 Modified MLP，主干先经过三角/多项式 Lift Layer，最终通过分支—主干点积输出坐标。训练目标为论文公式(14)--(16)、(21)--(22)的 PDE 残差平方与边界坐标误差平方加权和，不含内部网格监督项。每点对坐标分量求平方和，再对点与样本平均；默认 PDE/边界权重分别为 `1e-4`、`1`。

空间导数用中心有限差分，二维9点、三维19点；三维系数用度量矩阵的余子式，修正原文公式(18)中的下标和平方项不一致。残差、系数和差分操作全程保留参数梯度，不对预测值 detach。固定网格 MSE 仅用于监测；最佳模型和平台停止依据训练损失。

```powershell
python scripts/train/train_2d.py --model pimeshonet --dataset-mode both
python scripts/train/train_3d.py --model pimeshonet --dataset-mode both
python scripts/train/train_evaluate.py --dimension 2 --train-args "--model pimeshonet --dataset-mode both"
python scripts/evaluate/evaluate_2d.py --case esa_0_1000_20_20 --methods tfi,pde,meshonet,pimeshonet,hmonet
```

独立配置在 `pimeshonet_training`。四个模型全部训练使用 `--model all`；既有 `--model both` 的含义不变。完整命令及差分步长说明见 [训练评估命令说明](scripts/train/train_evaluate_cli.md#九pi-meshonet-自监督训练与评估)。

```text
configs/<维度>D/<数据集>/PI-MeshONet/<时间>/train.yaml
outputs/checkpoints/<维度>D/<数据集>/PI-MeshONet/<时间>/
outputs/training/<维度>D/<数据集>/PI-MeshONet/<时间>/
outputs/tensorboard/<维度>D/<数据集>/PI-MeshONet/<时间>/
outputs/results/<维度>D/<数据集>/<网格>/<参数>/PI-MeshONet/
```

checkpoint目录保存 `best.pt`、`best_model.pth`、`final.pt`、`final_model.pth` 及配置与摘要；训练日志含 `loss_pde`、`loss_boundary`。评估复用网格图、质量指标、原始边界误差、局部图及三维中间切片流程。


## Jacobian 损失计算方式

二维、三维监督 HMONet 在开启 Jacobian 损失时，默认使用 `hmonet_training.jacobian_method: finite_difference`，差分步长 `jacobian_fd_step: 0.001`。
原自动微分实现保留，可设置 `jacobian_method: autograd`，或使用命令行 `--jacobian-method autograd` 切换。
有限差分对最终硬约束坐标求导，内部采用中心差分，边界附近采用二阶单侧差分；所有差分查询点位于计算域内。
主坐标损失和残差正则仍只计算中心点。两种方式使用相同的正 Jacobian 惩罚定义，方法与步长随训练参数及摘要保存。
两个独立函数为 `jacobian_positive_loss_autograd`、`jacobian_positive_loss_finite_difference`，位于 `scripts/train/training_common.py`；旧的 `jacobian_positive_loss` 入口仍调用自动微分，以兼容既有调用。


### 二维高曲率与凹域测试边界

`src/data/geometry/boundary_generator_2d.py` 新增 `smooth_constricted_channel`（光滑相向凹陷）、`variable_width_s_channel`（变宽 S 通道）、`rounded_u_notch`（圆角深 U 槽），并完善 `gear_spline`（四边波纹样条）。这些边界均按下、右、上、左存储；上下从左到右，左右从下到上，参数域为 [0,1]。

Bézier、B 样条、直线和圆弧均复用解析导数与可微 Torch 求值，支持现有批量边界输入及字典序列化。U 槽采用弧长比例分配分段参数，连接点导数遵循两侧全局参数导数取平均的规则。圆弧与直线相切，但曲率在连接点处可以跳变。

波纹控制点数随齿数增长，避免原来的控制点欠采样；`amplitude` 表示控制点偏移，实际样条幅值会略有衰减。新增边界暂未配置独立的内插/外推数据集参数范围。

```powershell
python scripts/visualize/visualize_boundaries.py --names smooth_constricted_channel variable_width_s_channel gear_spline rounded_u_notch --nx 201 --ny 201
```

图片保存在 `outputs/boundaries/2D/`。采样包含每条边的两个端点，采用参数域均匀采样；默认21点仍可使用，但展示波纹和小圆角建议201点。均匀采样不额外插入非均匀的分段连接点，连接点参数可从分段曲线对象查询。


### 四类新数据集与边界参数可视化

`SELECTED_DATASETS` 当前选择 scc（光滑凹陷通道）、vsc（变宽 S 通道）、gsp（波纹样条）、run（圆角深 U 槽）的内插和外推，共8项。每项默认1000条；scc/vsc/run 默认41×41，gsp 默认81×81，几何参数范围见生成文件顶部 `DATASET_FAMILIES`。旧数据不会覆盖，已有同名目录会报错。

```powershell
python src/data/generation/make_dataset_parametric_2d.py --size 1000 --workers 8 --winslow-max-iter 10000 --winslow-tolerance 1e-6
python scripts/visualize/visualize_boundaries.py --nx 201 --ny 201 --surface-grid 21
```

并行求解采用有序返回，`dataset.npy` 与 `boundary_geometry.npy` 相同索引始终对应。PDE收敛状态与局部网格有效性分别记录；尤其深 U 槽可能收敛但存在局部翻转，不能直接视为合格真值。实际样本统计及全部数据集参数说明位于 `data/raw/2D/README.md`。

可视化路径为 `outputs/boundaries/<2D或3D>/<完整边界名称>/`，包含 `default.png`、各单参数修改图和 `parameters.json`。二维变参图叠加默认轮廓虚线。每幅图除文件名所示参数外，其余均取边界函数默认值，未必等于某个数据集的固定参数。



## 跨平台运行

Windows/Linux 的环境、路径迁移与已有实验复用说明见 [docs/cross_platform.md](docs/cross_platform.md)。

