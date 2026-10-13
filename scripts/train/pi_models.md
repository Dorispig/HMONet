# PI-MeshONet 与 PI-HMONet 使用说明

## 模型与输入输出

二维、三维均支持四种模型：`hmonet`、`meshonet`、`pimeshonet`、`pihmonet`。
`both` 保持只训练 HMONet 和 MeshONet；`all` 训练全部四种模型。每个数据集、模型分别训练与保存。

PI-MeshONet 使用独立的 PIMeshONet2D/3D。每个物理坐标对应一个 Adjust Layer 和门控 Modified MLP 分支，计算坐标经过三角/多项式 Lift Layer 与共享 Modified MLP 主干，分支和主干通过内积加偏置输出物理坐标。输入仍是边界采样点和计算域查询坐标，输出是物理域坐标。
PI-HMONet 复用 HMONet2D/3D，额外输入二维四条参数边界曲线或三维六个参数曲面，在网络内部施加硬边界约束。
二维查询坐标为 `(xi, eta)`，三维为 `(xi, eta, zeta)`，均位于单位计算域。
批量边界输入形状为 `[B, Nb, d]`，查询为 `[B, Nq, d]`，输出为 `[B, Nq, d]`。
网络内部按各坐标分量传参，训练辅助函数将输出拼接为上述张量。

PI-HMONet 的 `current` 直接使用参数几何投影点进行硬约束；`coons_residual` 使用精确 TFI 基础坐标加网络残差。
二维残差包络为 `16*xi*(1-xi)*eta*(1-eta)`；三维为 `64*xi*(1-xi)*eta*(1-eta)*zeta*(1-zeta)`。
包络在边界为零，因此输出符合给定参数边界。三维 TFI 的面、边和角点均由参数几何精确求值。

## 每轮训练过程

1. 按随机顺序遍历数据集；同一批次共享一组重新生成的均匀随机差分中心点。
2. 各轴步长为 h，中心限定在 `[h, 1-h]` 内，保证所有差分点均位于 `[0,1]`。
3. 二维每个中心构造 9 点差分模板，三维构造 19 点模板，包括混合偏导所需的点。
4. 将全部模板点与边界查询点拼接，整个批次一起前向计算；PI-HMONet 使用硬约束后的最终坐标。
5. 通过中心差分计算一阶、二阶和混合偏导，组装椭圆 PDE 残差。
6. 对模型参数执行常规反向传播与 Adam 更新；不对输入坐标调用自动微分计算偏导。
7. 按各批次样本数加权汇总每轮损失，更新公共学习率调度、日志、固定网格 MSE 监测和收敛判断。

真实内部 PDE 网格坐标不进入这两个模型的训练目标。目前仍复用项目的数据集读取器，数据文件中保留真实网格，供固定网格 MSE 监测及训练收尾统计使用。
因此这是优化目标自监督，并不意味着当前数据文件已经改成不含网格标签的边界专用格式。

## 损失公式

令 r 是网络最终输出的二维或三维坐标向量。一阶、纯二阶和混合二阶偏导使用：

```text
r_i  ≈ (r(q+h_i*e_i) - r(q-h_i*e_i)) / (2*h_i)
r_ii ≈ (r(q+h_i*e_i) - 2*r(q) + r(q-h_i*e_i)) / h_i²
r_ij ≈ (r++ - r+- - r-+ + r--) / (4*h_i*h_j)
```

二维：`alpha = r_eta·r_eta`，`beta = r_xi·r_eta`，`gamma = r_xi·r_xi`。
残差向量为 `R = alpha*r_xixi - 2*beta*r_xieta + gamma*r_etaeta`。
三维：令 `G_ij = r_i·r_j`，`C = adj(G)`，则 `R = Σ C_ii*r_ii + 2*Σ(i<j) C_ij*r_ij`。
三维实现使用叉积构造度量余子式，避免求逆；没有照抄论文式(18)中不一致的下标及缺失平方。

PDE 损失为 `L_PDE = mean(||R||²)`，边界损失为 `L_B = mean(||r_boundary - exact_boundary||²)`。
坐标分量先求平方和，再对采样点和批次求平均；不是先对坐标分量取平均，也不额外统一乘以维度。

- PI-MeshONet：`L = lambda_PDE * L_PDE + lambda_B * L_B`，默认权重 `1e-4` 和 `1.0`。
- PI-HMONet：默认 `L = L_PDE`，默认权重为 `1.0`。始终记录边界损失，但默认不加入优化。
- PI-HMONet 的参数曲线硬约束直接保证边界，因此不计算边界损失。
- 不自动继承监督 HMONet 的 Jacobian、法向或残差正则损失，以保持当前 PDE 自监督目标明确。

PI-HMONet 默认权重为 1，因此总训练损失就是 PDE 残差平方和的均值；训练损失阈值 `1e-6` 直接作用于该 PDE 损失，不等于坐标 MSE 阈值。仍保留权重配置以支持实验或可选边界损失。修改权重会改变损失与梯度尺度，历史训练配置归档不随默认值修改。

## 收敛、模型选择和输出

四种模型使用同一个 `TrainingState` 和同一套平台收敛函数。
训练损失连续达标、平台相对/绝对波动、接近历史最佳、连续稳定和无显著改善均复用公共实现。
两种 PI 模型固定使用训练损失选择最佳模型与判断平台；关闭基于真实网格 MSE、最大点误差和 P99 的停止条件。
MSE 可按原配置频率记录，但不会决定 PI 模型的权重更新、最佳模型或停止轮次。
训练损失精度条件独立于平台条件，满足其连续轮数就可提前结束。

```text
outputs/checkpoints/<2D或3D>/<数据集>/<划分>/PI-HMONet/<映射>/<日期_时分>/
  best.pt, best_model.pth, final.pt, final_model.pth
  config.json, config.yaml, checkpoint_metadata.json, model_parameters.json
  training_summary.json, training_summary.csv
outputs/training/<2D或3D>/<数据集>/<划分>/PI-HMONet/<映射>/<日期_时分>/
  训练日志、损失/MSE数据与曲线、训练摘要、TensorBoard（开启时）
outputs/tensorboard/<2D或3D>/<数据集>/<划分>/PI-HMONet/<映射>/<日期_时分>/
configs/<2D或3D>/<数据集>/PI-HMONet/<映射>/<日期_时分>/train.yaml
```

PI-MeshONet 使用对应 `PI-MeshONet` 目录，不含映射层级。
评估结果位于 `outputs/results/<维度>/<数据集>/<网格规模>/<参数值>/PI-HMONet/<映射>/<raw或后处理方式>/`。
默认后处理为 `none`，同时保留原始边界误差、计时、网格和所选质量数据/图。
方法对比、局部图和三维三个中间切片沿用现有评估流程；缓存按模型检查点、参数和映射区分。

## 命令行

在项目根目录执行（先激活 MGNet 环境）。网格规模必须与所选数据集一致。

```powershell
# 按二维 train.yaml 训练 PI-HMONet
python scripts/train/train_2d.py --model pihmonet
# 三维 Coons 自监督训练
python scripts/train/train_3d.py --model pihmonet --pihmonet-mapping-mode coons_residual
# 四种模型分别训练
python scripts/train/train_2d.py --model all
# PI-HMONet 专属覆盖；同名专属参数优先于 --pi-* 公共覆盖，最后使用 YAML
python scripts/train/train_2d.py --model pihmonet --pihmonet-mapping-mode current --pihmonet-pde-loss-weight 1 --pihmonet-interior-sample-count 128 --pihmonet-fd-step 0.02 0.02
# PI-MeshONet 对应专属参数
python scripts/train/train_2d.py --model pimeshonet --pimeshonet-pde-loss-weight 1e-4 --pimeshonet-boundary-loss-weight 1 --pimeshonet-interior-sample-count 128 --pimeshonet-fd-step 0.02 0.02
# 评估指定模型（路径替换成实际检查点）
python scripts/evaluate/evaluate_2d.py --case rse_ext_0_1000_20_20 --methods tfi,pde,pihmonet --pihmonet-checkpoint outputs/checkpoints/2D/rse_ext_0_1000_20_20/PI-HMONet/current/<日期_时分>/best.pt --pihmonet-mapping-mode current --pihmonet-boundary-projection none
# 自动选择对应数据集最新检查点；both 同时评估两种 PI-HMONet 映射
python scripts/evaluate/evaluate_2d.py --case rse_ext_0_1000_20_20 --model pihmonet --pihmonet-mapping-mode both
# 串联训练与评估，自动传递本次检查点和映射
python scripts/train/train_evaluate.py --dimension 2 --train-args "--model pihmonet --pihmonet-mapping-mode current"
```

`--hmonet-mapping-mode` 只覆盖 HMONet，`--pihmonet-mapping-mode` 只覆盖 PI-HMONet。
评估还支持 `--pihmonet-checkpoint`、`--pihmonet-boundary-projection none/direct/inverse_distance/idw`。
`--checkpoint` 仍只适用于恰好选择一种网络的情况。评估 `--model all` 选择四种网络；`--methods` 优先。

## 与论文的对应与差异

依据用户提供的《A generalizable and self-supervised method for structured ...》PI-MeshONet 论文第3.2、3.3节：

| 内容 | 当前实现 |
| --- | --- |
| 不用内部坐标标签监督 | 两种 PI 模型均符合 |
| PDE 与边界平方残差，式(14)-(16)、(21)-(22) | PI-MeshONet 对应；PI-HMONet 默认用硬约束替代边界罚项 |
| 中心有限差分，式(23)-(27) | 对应；三维采用19点模板实现同类混合偏导 |
| 二维椭圆方程，式(15) | 对应 |
| 三维椭圆方程，式(17) | 使用标准度量余子式；修正式(18)的明显不一致，并有解析函数数值测试 |
| 网络结构 | PI-MeshONet已采用论文Adjust/Modified-MLP分支/Lift-Modified-MLP共享主干/点积输出拓扑；论文未公开全部层宽，因此层宽与24,452参数量不宣称完全复现 |
| PI-HMONet 硬约束 | 本项目扩展，论文原模型没有此硬约束结构 |
| 采样、步长、优化与停止 | 使用本项目配置；中心点排除宽度 h 的近边界带，非完整论文实验复刻 |
| MSE 监测 | 使用当前数据集的 PDE 网格；不影响自监督训练决策 |

因此 PI-MeshONet 已对齐论文公开的网络拓扑、自监督目标与有限差分方法；未公开的层宽、优化细节以及论文数据集仍不属于逐项复现。
PI-HMONet 是在该自监督方法上增加精确参数边界硬约束的扩展。
小规模测试验证可运行、梯度与边界正确性，不代表已完成全部数据集的长期收敛和网格质量验证。
