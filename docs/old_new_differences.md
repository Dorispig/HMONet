# 原项目与 HMONet_p 当前训练/评估差异

以下差异是按当前代码逐项核对得到的，后续应以本文件作为迁移验收清单。

## 训练

1. 原项目由 `src/engine/trainer.py` 负责完整训练循环；新项目由 `scripts/train/train_2d.py` 和 `scripts/train/train_3d.py` 分维度负责。
2. 原项目数据加载器直接读取旧版边界点、规则网格和 PDE 网格字段；新项目读取 `geometry.json`、`boundary_samples.npy`、`physical_samples.npy`、`pde_mesh.npy` 和 `sample_parameters.npy`。
3. 原项目支持 adaptive、non_adaptive 和 both 三种 HMONet sampling；新项目已删除 sampling 配置和命令行，固定使用均匀随机采样。
4. 原项目二维 HMONet 损失由固定网格损失、随机点损失、随机点权重、PDE Winslow 损失、Jacobian 约束、边界法向约束和 residual 正则组成；新项目目前实现了随机点中心权重及随机点损失倍率，Jacobian、法向、Winslow PDE 和 residual 项尚未全部接入。
5. 原项目 MeshONet 使用 interior/boundary 加权损失；新项目当前使用统一坐标 MSE。
6. 原项目支持完整的 MSE 收敛、训练损失收敛、plateau、最佳改进比例和早停状态机；新项目目前保存 best/final，但训练循环尚未恢复全部早停状态机。
7. 原项目输出详细的 loss component、dataset MSE history、TensorBoard 和完整 `training_summary.json` 字段；新项目已输出基本 `training_summary.json`，但字段数量仍少于原项目。
8. 原项目 checkpoint 可按模型、采样模式、映射方式和数据集模式分层组织；新项目按 `<dataset-mode>/<model>` 组织，映射方式尚未独立分目录。
9. 原项目支持训练时从 YAML 和命令行覆盖大量超参数；新项目目前覆盖 config、dataset mode、model、epochs、batch size，其他高级训练参数尚未全部开放。

## 评估

1. 原项目评估入口使用完整的 `evaluate.py`/`evaluate_3d.py` 配置 cases；新项目使用简化的 `evaluate_2d.py` 和 `evaluate_3d.py` 单数据集入口。
2. 原项目默认比较 TFI、PDE、MeshONet、HMONet 的多个 mapping/sampling 变体；新项目已固定生成四个方法目录，但缺失 checkpoint 时会用 TFI 兼容占位结果。
3. 原项目 PDE 结果来自实际 Winslow 求解；新项目当前 PDE 评估仍使用参数曲线 TFI 初始化结果，需继续接入正式 PDE 迭代。
4. 原项目质量输出器保存完整网格、质量字段、summary、CSV、PNG/PDF 和指标定义；新项目已接入 `save_quad_quality`/`save_hex_quality`，但部分比较级汇总文件仍少于原项目。
5. 原项目质量图使用统一质量方向、离散颜色映射、全局 colorbar 和所有质量指标；新项目已调用原质量输出器，三维切片也调用二维质量输出器，但需要对实际数据逐项确认色标范围。
6. 原项目记录非法单元数量及比例，包括凹单元、翻转单元、退化单元和自相交；新项目三维自相交表目前仍是占位值，尚未接入完整检测器。
7. 原项目局部图支持按坐标范围、worst 指标、参考方法、cell padding、全局/局部色标和方法分组；新项目二维当前支持坐标范围，worst 自动定位和完整 3D 局部参数尚未全部接入。
8. 原项目三维输出边界面和中间面质量图；新项目已创建 x_mid、y_mid、z_mid 方法目录并保存网格和质量数据，但中间面比较图的布局与原项目仍需校准。
9. 原项目保存计时、网格生成时间、网络推理时间和 paper timing table；新项目当前评估尚未完整恢复这些计时表。
10. 原项目支持缓存和增量复用质量分析结果；新项目目前每次评估重新计算。
11. 原项目支持直方图开关和 PDF/PNG 双格式；新项目默认关闭直方图，并主要输出 PNG。

## 数据和几何

1. 原项目边界由离散边界数组描述；新项目保存参数曲线/参数曲面及其均匀采样点。
2. 新项目部分复杂二维、三维边界生成器仍为迁移占位实现，尚未达到原项目所有数据集几何完全等价。
3. 新项目的 `physical_samples.npy` 兼容旧式物理采样数组；当文件不含逻辑坐标时训练脚本会临时生成均匀随机逻辑点。

