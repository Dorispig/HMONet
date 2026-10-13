# 新旧项目评估对比：air_t

> 本文为修复前的检查记录。评估流程已随后重写，当前功能、差异和输出结构见 [评估体系与输出说明](evaluation_system.md)。下文的“当前”指检查当时的旧实现。

检查日期：2026-09-15。旧项目：`E:/projects/HMONet`；新项目：`E:/projects/HMONet_p`。

结论：**当前新项目评估没有与旧项目对齐，且有影响结果真实性的问题。二维正常退出并不代表评估正确，三维当前不能完成完整评估。** 本次修改二维评估配置并进行审查、试跑，没有修改评估实现、模型权重或正式评估结果。

## 1. 本次配置修改

文件：`configs/2D/evaluate.yaml`。

| 项目 | 设置 |
| --- | --- |
| 默认 cases | `air_t_int_0_1000_21_20`、`air_t_ext_0_1000_21_20` |
| 内插测试 t | 0.275、0.30、0.325 |
| 外推测试 t | 0.62、0.65 |
| 训练网格 | 21×20 |
| 评估网格 | 21×20、51×50、101×100 |
| 固定几何参数 | m=0.02、p=0.4、c=1.0、radius=1.0、center=(0,0) |
| 参数边界工厂名称 | `circle_airfoil`；变化参数单独指定为 `t` |
| 模型/方法选择 | `model: both`；HMONet、MeshONet、TFI、PDE |
| HMONet 内部投影 | current |
| 两个模型边界后处理 | direct |
| PDE 配置 | 最大迭代 1000、容差 1e-8 |
| 期望计时配置 | 预热 1 次、正式推理 3 次，取中位数 |
| 期望绘图配置 | PNG 开启、PDF 关闭、分布图关闭 |

测试 t、训练网格、评估网格、训练区间均与旧项目对应 air_t case 核对一致。内插训练区间为 `[0.05,0.25] ∪ [0.35,0.60]`；外推训练区间为 `[0.05,0.60]`。默认选择由旧配置的仅外推改为两组 air_t；其他 case 模板保留，但不作为默认对象。移除了不适用于新项目的 `hmonet_sampling_mode` 配置。

**配置尚未生效于评估入口**：当前 `scripts/evaluate/evaluate_2d.py` 不读取 YAML，甚至没有 `--config` 参数；`--model` 只接受一个模型。此次已在 YAML 开头明确标注这一限制。上述两组共 15 个“参数值×评估网格”任务，当前入口还不能直接按 YAML 批量执行。

已有两个训练数据集不包含上述五个测试厚度。当前入口的 `--parameter-index` 只能选择保存的数据集样本，不能用索引选到这些留出的测试参数。需要后续接入由 case 参数直接构造精确边界的评估流程，不能把训练样本上的输出称为这些参数的内插/外推测试结果。

## 2. 二维评估逐项对比

| 项目 | 旧项目 | 新项目当前实现 | 结论 |
| --- | --- | --- | --- |
| 入口与配置 | `scripts/evaluate.py` 读取 evaluate/train YAML，支持 CLI 覆盖 | `scripts/evaluate/evaluate_2d.py` 只读取少量 CLI 参数 | 未对齐 |
| 测试对象 | case、指定几何参数值，可遍历多个 case | 已保存数据集中的一个索引，默认第 0 条 | 未对齐 |
| 多分辨率 | 区分训练边界输入分辨率与查询网格分辨率 | 输入和查询都使用数据集保存的尺寸 | 未接入跨分辨率评估 |
| 几何来源 | 离散边界生成器 | 从 `boundary_geometry.npy` 恢复参数边界类 | 按新设计保留的差异 |
| 模型组合 | 支持两个模型同时运行及选择方法 | 一次只接受 hmonet 或 meshonet | 未对齐 |
| 自动选择 checkpoint | 按模型、数据集、投影/采样配置寻找 | 扫描数据集目录下全部 best.pt，未按请求的模型和映射过滤 | 可能选错模型 |
| checkpoint 加载 | 从模型参数信息恢复结构 | 2D 使用 checkpoint.common/hmonet 配置，未合并 meshonet 专属配置；裸 state_dict 默认当作 HMONet | 并非所有 checkpoint 格式均适配 |
| 缺失模型 | 按配置跳过缺失模型及依赖它的对比组 | 用 TFI 复制网格填充 HMONet/MeshONet | 错误结果会带上模型名称 |
| 方法名称 | 各方法使用统一标识 | 预测使用 HMONET/MESHONET，占位使用 HMONet/MeshONet | 重复标签，Windows 路径覆盖 |
| HMONet 输入 | 旧网络的离散边界输入 | 已传入参数边界类 `boundary_geometry` | 已接入新模型输入 |
| HMONet 内部投影选择 | current/coons_residual/both 可配置 | 主要来自 checkpoint；无评估 CLI 切换或双映射循环 | 未对齐 |
| 查询端点 | 旧 2D 将查询点夹至域内，再恢复物理边界 | 使用含 0/1 的完整计算域网格 | 新精确边界设计下可以保留，但不是完全相同 |
| 边界后处理 | 两个模型均支持 direct/IDW，保存变体 | CLI 已接入两种后处理 | 功能部分具备 |
| IDW 幂次 | YAML/实现可使用指定幂次 | 调用默认 2.0，未读取 YAML | 参数未接入 |
| 二维 TFI | 离散边界 TFI | 参数曲线 TFI | 按新设计保留的差异 |
| PDE 参数 | 使用评估配置 | 固定调用求解器默认值：1000 次、1e-6 | YAML 的 1e-8 未生效 |
| PDE 失败 | 按求解流程及异常处理报告 | 捕获任何异常后复制 TFI，仍命名 PDE_Winslow；只在 timing 标志中注明 | 不能将替代结果当 PDE 成功结果 |
| PDE 收敛记录 | 有较完整的生成元信息 | 评估未请求求解器 `return_info`，不记录迭代数或是否收敛 | 缺失 |
| HMONet 后续 PDE 平滑 | `hmonet_pde_iterations` 可用 | 没有调用 | 未接入 |
| 推理设备 | 配置/CLI 可选 | 2D 自动 CUDA 或 CPU，无覆盖入口 | 未对齐 |
| 查询分块 | `query_batch_size` 控制 | 整个网格一次前向 | 未接入分块 |
| 推理计时 | 预热、重复取中位数；区分网络/边界后处理/总耗时 | 虽有 start 变量，但未输出网络计时；仅记录整个评估耗时 | 不能用于网络速度比较 |
| 原始模型网格 | 保存后处理前 `none` 结果及其质量、边界误差 | 预测后直接覆盖为后处理结果，无 raw/none 产物 | 缺失 |
| 边界误差 | 对后处理前预测计算逐点误差并保存数据、摘要、图 | 只计算后处理后的最大/平均误差，放入 comparison CSV | 不能衡量网络本身的边界精度 |
| 网格 MSE | 对当次 Winslow 网格输出逐点、逐单元误差和摘要/图 | 只计算一个全网格标量，相对数据集中存储的目标网格 | 参考网格、明细输出不同 |
| MSE 坐标系数 | 旧评估逐点误差是坐标分量取平均 | 新评估也是坐标分量取平均，未乘 2 | 此项不是新旧差异；但与当前训练固定网格 MSE×2 口径不同 |
| 单方法网格图 | 网格连线图 | `grid_quality/mesh.png` 有网格连线图 | 已具备 |
| 单方法质量指标 | 13 个二维指标及逐单元数值 | 13 个指标已调用同一迁移的质量输出逻辑 | 主体已迁移 |
| 非法单元比例 | 综合非法单元 fraction；各标志有统计 | 同样有综合 fraction；各标志仅 count/total，未逐项单独保存比例 | 不应宣称各类比例都已实现 |
| 四方法质量对比 | 调用 `save_quality_comparison`，统一色标，逐指标对比图和数据 | 没有调用该函数；仅 scatter 对比，复制各方法目录 | 核心对比未接入 |
| comparison 表格 | 质量指标的详细统计 | `quality_table.csv` 只有 MSE、边界最大/平均误差、坐标范围 | 文件名不能代表完整质量表 |
| 分布图开关 | comparison 支持；旧单方法图自身也包含分布图 | 新单方法图仍固定“分布图+颜色图”，不读取 YAML | 用户要求关闭分布图仍未完成，旧项目也不能作为全部符合要求的基准 |
| 颜色规范 | 单方法 jet/jet_r、面积特定色图；comparison 共用色标 | 单方法对应逻辑已迁移；跨方法共用色标未接入 | 仅部分一致 |
| 局部图 | 独立脚本支持物理坐标、质量指标、worst、读取现有缓存 | 只有 `--local x0 x1 y0 y1` 的散点截取，目录硬编码为 local/mse | 没有质量着色、worst 选择与独立重绘流程 |
| 结果复用 | 生成/分析分别校验签名、checkpoint、参数和文件完整性 | 只检查参数目录的 `.complete` | 换模型、权重、投影、局部范围也可能被直接跳过 |
| 强制执行 | 分别强制生成/分析等 | 单个 `--force`，无分阶段控制 | 未对齐 |
| 输出目录 | 参数值如 `0.65`；模型区分映射及后处理 | 参数标签 `t_0.65`；后处理仅在 comparisons 路径区分 | 不完全符合此前要求的路径/变体隔离 |
| 结果摘要与论文表 | results/evaluation_summary/timings、生成缓存、点误差等；论文表依比较组生成 | 只保存简化 parameters/timing 和 comparison 表 | 缺失多项 |
| 运行进度 | case、参数、网格、方法、分析阶段进度 | 最后输出 results 路径；偶有 fallback/skip 信息 | 未对齐 |
| 相对路径 | 多处按项目 ROOT 解析 | dataset/output/checkpoint 自动寻找依赖当前工作目录 | 从其他目录运行的行为不同 |

关于自适应采样：旧二维有自适应/非自适应组合及相互比较；新项目按用户要求取消自适应，这部分应有意简化，不需要为了“相同”重新加入。参数边界类输入、参数曲线 TFI 和新 checkpoint 目录也应保留。

## 3. 二维实际试跑及文件

环境：MGNet，当前 CUDA 设备；输入现有 `air_t_int_0_1000_21_20` 第 0 条，t=0.05，21×20；指定已训练 HMONet current 的 `20260915_2011/best.pt`。

试跑输出在临时目录 `C:/Users/24846/AppData/Local/Temp/hmonet_evaluate_audit_q4a8_3zn/2D/`，没有覆盖 `outputs/results`。这次是评估流程检查，**不是上述留出参数的泛化测试**。

- 进程退出码为 0，共生成 226 个文件，包含大量复制文件。
- 表格内的方法为 `TFI, PDE_Winslow, HMONET, MeshONet, HMONet`，不是四个不同的真实方法。
- `HMONet/mesh.npy`、`HMONet/current/mesh.npy`、`MeshONet/mesh.npy` 均与 TFI 完全相同。
- Windows 不区分大小写：真实预测写入 HMONET，随后 HMONet 的 TFI 占位再次写入同一路径。`HMONet/mesh.npy` 与同目录 `grid_quality/mesh_data.npz` 内网格的最大坐标差为约 **0.127205**，网格文件与质量数据已经不对应。
- comparison 图实测是五栏散点图：一栏真实 HMONET、另外两栏以模型名字标记的 TFI 占位。
- 本次总评估耗时约 42.57 s，包含求解、绘图和文件写入；**不是网络预测耗时**。
- 当前查到两个 air_t HMONet best.pt；未查到 MeshONet best.pt。缺失模型本应明确跳过，不能以 TFI 代替。

在默认输出根目录下，当前二维入口成功运行时的文件形式如下（这里展示实际代码行为，错误内容已注明）：

```text
outputs/results/2D/<dataset>/<nx>x<ny>/t_<value>/
├── evaluation_parameters.json
├── evaluation_timing.json
├── .complete
├── TFI/
│   ├── mesh.npy
│   └── grid_quality/                    # 公共质量文件见下方
├── PDE_Winslow/
│   ├── mesh.npy
│   └── grid_quality/
├── MeshONet/
│   ├── mesh.npy                         # 本次为 TFI 占位
│   └── grid_quality/
├── HMONet/
│   ├── mesh.npy                         # 本次被 TFI 占位覆盖
│   ├── grid_quality/                    # 本次真实 HMONET 的质量文件
│   └── current/
│       ├── mesh.npy                     # 本次为 TFI 占位
│       └── grid_quality/
└── comparisons/
    ├── quality_summary.json             # 只有 methods 和 mse
    ├── quality_table.csv                # 非完整网格质量表
    └── current/hmonet_direct__meshonet_direct/
        ├── comparison.png              # 散点图，本次有重复方法
        ├── <各方法复制的网格与质量文件>/ # Windows 大小写冲突导致覆盖
        └── local/mse/comparison.png     # 仅指定 --local 时生成
```

每个二维 `grid_quality` 目录生成：

```text
mesh_data.npz
mesh.png
quality_data.npz
quality_data.csv
quality_summary.json
quality_<metric>.png                    # 每张含分布统计和质量颜色图
invalid_cells.json
invalid_cells.csv
invalid_cells_visualization.png
```

13 个 `<metric>` 为 `max_angle`、`min_angle`、`jacobian`、`scaled_jacobian`、`skewness`、`aspect_ratio`、`orthogonality`、`cell_area`、`concave_vertex_count`、`self_intersection`、`intersecting_cell`、`flipping_cell`、`degenerate_cell`。

入口**不生成**原项目完整的 raw/none 模型结果、逐点误差、逐单元误差、统一色标四方法质量对比图/数据、网络推理时间表，以及完整生成/分析缓存。

## 4. 三维检查

air_t 是二维 case，所以本次没有修改三维 YAML。三维入口具有上述许多同类问题，并存在独立错误：

| 项目 | 当前问题 / 与旧项目差异 |
| --- | --- |
| 完整运行 | 临时 3×3×3 box 数据实际试跑，在 `evaluate_3d.py:241` 调用未导入的 `save_quad_quality`，抛出 NameError，不能完成输出 |
| 查询点排列 | `meshgrid(x,y,z,indexing='xy')` 展平与后续 `(nz,ny,nx,3)` 解释不一致；旧项目按一致的逻辑索引生成/分析 |
| 切片数据形状 | 后续将带三个坐标分量的三维数组直接送入二维质量接口，尚未形成正确的切片二维坐标和扁平网格 |
| 精确边界误差 | `evaluate_faces` 返回六个面各自的投影数组，形状为 `(6,N,3)`；入口把它当 `(N,3)` 精确网格并按边界节点索引，维度错误 |
| 自相交对比表 | 后续代码将每种方法的 self_intersection 直接写为 0，没有检测；不能当作无自相交证据 |
| surface_quality_table.csv | 后续代码只写 method 一列，没有质量指标 |
| 设备/计时 | 模型在 CPU 上，没有设备选择、预热/重复计时或网络单独耗时 |
| 整体三维图 | 仅调用 save_hex_quality，且关闭其分布图；没有接入旧项目整体网格图、表面/体质量着色图的完整路径 |
| 中间切片 | 意图创建 x_mid/y_mid/z_mid，但运行在第一张切片处中断；不是已经能完整输出三组切片及对比表 |
| 局部图 | `zoom_3d.py` 辅助功能文件存在，但新评估入口没有调用或 CLI 接口；不能把“文件已复制”当成“功能已接入” |
| 3D TFI 数值正确性 | 当前 tfi_from_surfaces 在标准 box 上也错误：计算域 (0,0,0) 应输出 (-1,-1,-1)，实际 (0,0,0)；(1,1,1) 应输出 (1,1,1)，实际 (2,2,2) |
| 数据集参考网格 | 新三维生成器仍将 TFI 结果保存到 pde_mesh.npy，metadata 也写 tfi_initial；不能据文件名认为是 PDE 真值 |

三维试跑只留下部分 TFI 质量数据和 x_mid/mesh.npy，尚未生成完整的 parameters/timing、comparison 表、完成标记。后续代码里出现的 volume_quality_table.csv、surface_quality_table.csv、self_intersection_table.csv、z_mid.png，目前不能称为经过本次验证的实际完成产物。

## 5. 验证范围与修复优先级

已完成：新 YAML 可解析；五个测试值和三种网格与旧 air_t 配置一致；当前参数边界工厂可以构造五个测试边界；二维现有 checkpoint 完整试跑与落盘数据核对；三维 box 最小试跑；三维 TFI 的已知解析答案检查。详见同目录 `evaluation_audit_evidence.json`。

此前测试主要覆盖训练、边界及评估入口的 `--help` 等；帮助命令成功不能证明完整评估正确。本次没有把“退出码为 0”当作正确性判据，也没有宣称已经验证所有几何类型和所有模型配置。

后续建议按以下顺序修复：

1. 去除 TFI 模型占位、统一方法名称，修复 Windows 覆盖和 checkpoint 按模型筛选。
2. 接入 YAML/case、精确测试参数边界、训练输入尺寸与评估尺寸分离、both 模型运行。
3. 保存 raw/后处理两套结果，复用旧质量比较流程，恢复真正的边界误差、局部图和分阶段计时/缓存。
4. 修正 3D TFI、索引/切片/边界误差，接入真实自相交检测，再完成三维端到端评估。

本次仅配置修改和审查，这些实现问题仍待修复。
