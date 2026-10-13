# HMONet 二维训练架构与实现说明

本文档依据当前项目代码与 `41×20` air_t 训练配置整理，可作为 SCI 论文中 Methodology、Training Strategy 与 Implementation Details 部分的基础材料。

主要实现文件：

- `src/models/model.py`：HMONet 网络及边界硬约束；
- `src/models/modules.py`：MLP 等基础网络模块；
- `scripts/train/train_2d.py`：二维 HMONet 完整训练流程；
- `scripts/train/training_common.py`：采样、目标插值与损失函数；
- `scripts/train/training_state.py`：收敛判断、日志及模型保存；
- `configs/2D/train.yaml`：当前二维训练参数。

## 1. 问题定义

HMONet 学习从参数化物理边界到结构网格坐标的算子映射：

$$
\mathcal{G}_{\theta}:(\mathbf B,\Gamma,\boldsymbol{\xi})\mapsto\mathbf x,
$$

其中，$\mathbf B\in\mathbb R^{N_b\times2}$ 为均匀采样边界点，$\Gamma=\{\Gamma_l,\Gamma_r,\Gamma_b,\Gamma_t\}$ 为四条精确参数边界曲线，$\boldsymbol{\xi}=(\xi,\eta)\in[0,1]^2$ 为计算域坐标，$\mathbf x=(x,y)\in\Omega$ 为预测物理坐标。

对于当前 $41\times20$ 网格：

$$
N_b=2(N_x+N_y)-4=118.
$$

每条训练数据包含精确参数边界、118 个均匀边界采样点和 Winslow PDE 迭代网格。内插集与外推集分别训练、评估和保存模型。

## 2. 网络整体结构

HMONet 包含边界分支网络、计算坐标主干网络和精确边界硬约束映射：

$$
\widehat{\mathbf x}_{\mathrm{raw}}=\mathcal N_\theta(\mathbf B,\boldsymbol{\xi}),
$$

$$
\widehat{\mathbf x}=\mathcal H(\widehat{\mathbf x}_{\mathrm{raw}},\Gamma_l(\eta),\Gamma_r(\eta),\Gamma_b(\xi),\Gamma_t(\xi),\boldsymbol{\xi}).
$$

网络学习内部映射，参数曲线直接提供边界投影坐标。

## 3. 边界分支网络

边界横、纵坐标分别表示为 $\mathbf p_x,\mathbf p_y\in\mathbb R^{N_b}$，并经过相互独立的调整网络：

$$
\mathbf a_x=A_x(\mathbf p_x),\qquad \mathbf a_y=A_y(\mathbf p_y).
$$

当前调整网络参数为：输入维数 118，4 个宽度为 128 的隐藏层，输出维数 64，激活函数为 Tanh。

调整特征进一步进入两个独立分支：

$$
\mathbf h_x=B_x(\mathbf a_x),\qquad\mathbf h_y=B_y(\mathbf a_y),
$$

每个分支具有4个宽度为128的隐藏层，输出维数为128。拼接后：

$$
\mathbf h=[\mathbf h_x,\mathbf h_y]\in\mathbb R^{256}.
$$

两个融合 MLP 分别生成用于预测 $x$ 和 $y$ 的分支向量。融合 MLP 的输入维数为256，包含4个宽度为128的隐藏层，输出维数为128。网络还从联合边界特征生成二维偏置和四个边界投影权重参数。

## 4. 计算坐标主干网络

查询坐标采用增强特征：

$$
\Phi(\xi,\eta)=[\sin\xi,\sin\eta,\cos\xi,\cos\eta,\xi,\eta,\xi^2,\eta^2,\ldots,\xi^6,\eta^6].
$$

主干网络输入维数为16，具有4个宽度为128的隐藏层，输出维数为128，激活函数为 Tanh：

$$
\mathbf t=T(\Phi(\xi,\eta)).
$$

## 5. 原始网络预测

分支特征与主干特征通过内积融合：

$$
\widehat{x}_{\mathrm{raw}}=\sum_{k=1}^{128}b_{x,k}t_k+c_x,
$$

$$
\widehat{y}_{\mathrm{raw}}=\sum_{k=1}^{128}b_{y,k}t_k+c_y.
$$

原始预测不单独承担边界拟合任务，最终输出还要经过精确参数边界硬约束。

## 6. 精确参数边界硬约束

当前正式配置使用 `current` 映射。对于任意查询点：

$$
\mathbf P_l=\Gamma_l(\eta),\quad\mathbf P_r=\Gamma_r(\eta),\quad
\mathbf P_b=\Gamma_b(\xi),\quad\mathbf P_t=\Gamma_t(\xi).
$$

投影点由参数曲线的 Torch 向量化公式直接计算，不进行离散曲线拟合。网络从联合边界特征预测：

$$
\rho_i=\operatorname{sigmoid}(W_i\mathbf h+c_i),\qquad i\in\{l,r,b,t\}.
$$

令 $a=\xi(1-\xi)$，$b=\eta(1-\eta)$，则：

$$
\omega_l=\frac{\rho_l(1-\xi)\eta(1-\eta)}{a+\rho_l\eta(1-\eta)},
$$

$$
\omega_r=\frac{\rho_r\xi\eta(1-\eta)}{a+\rho_r\eta(1-\eta)},
$$

$$
\omega_b=\frac{\rho_b\xi(1-\xi)(1-\eta)}{b+\rho_b\xi(1-\xi)},
$$

$$
\omega_t=\frac{\rho_t\xi\eta(1-\xi)}{b+\rho_t\xi(1-\xi)},
$$

$$
\omega_0=1-\omega_l-\omega_r-\omega_b-\omega_t.
$$

最终输出为：

$$
\widehat{\mathbf x}=\omega_0\widehat{\mathbf x}_{\mathrm{raw}}+\omega_l\mathbf P_l+\omega_r\mathbf P_r+\omega_b\mathbf P_b+\omega_t\mathbf P_t.
$$

该构造在边界上退化为对应参数曲线，因此：

$$
\widehat{\mathbf x}(0,\eta)=\Gamma_l(\eta),\quad
\widehat{\mathbf x}(1,\eta)=\Gamma_r(\eta),
$$

$$
\widehat{\mathbf x}(\xi,0)=\Gamma_b(\xi),\quad
\widehat{\mathbf x}(\xi,1)=\Gamma_t(\xi).
$$

因此 HMONet 不需要显式边界坐标损失，边界误差主要受浮点精度影响。

## 7. Coons 残差备选映射

代码还保留 `coons_residual` 模式：

$$
\widehat{\mathbf x}=\mathbf x_{\mathrm{Coons}}+E(\xi,\eta)\mathbf r_\theta,
$$

其中：

$$
E(\xi,\eta)=16\xi(1-\xi)\eta(1-\eta).
$$

由于 $E=0$ 在全部边界上成立，残差不会改变边界。当前正式训练使用 `current`，Coons 残差模式用于映射形式消融。

## 8. 训练点采样

当前正式配置采用均匀全局随机采样，每个 batch 生成400个计算域内部点：

$$
\boldsymbol{\xi}_j\sim U((0,1)^2).
$$

同一 batch 中所有几何样本共享同一组随机点，从而将整个 batch 和所有查询点一次性送入网络。DataLoader 设置为随机遍历、`batch_size=64`、`shuffle=True`、不丢弃最后一个 batch，随机种子为 `20260802`。

代码也支持曲率与边界距离自适应采样，但当前配置为 `uniform`。

## 9. 全局边界位移 IDW 监督目标

### 9.1 初始双线性插值

首先从离散 PDE 网格得到：

$$
\mathbf x_{\mathrm{bil}}(\boldsymbol{\xi})=I_{\mathrm{bil}}(\mathbf X_{\mathrm{PDE}},\boldsymbol{\xi}).
$$

### 9.2 边界加密采样

每个边界网格参数区间被四等分，只采用每段内部的 $1/4$、$1/2$、$3/4$ 三个新增点。原始边界节点已经精确，因此不再重复进入 IDW 支撑点集合。第 $i$ 段中的支撑参数为：

$$
t_{i,k}=\frac{i+k/4}{N-1},\qquad k=1,2,3.
$$

每条横向、纵向边界的支撑点数分别为：

$$
N_h=3(N_x-1),\qquad N_v=3(N_y-1),
$$

四条边界的总支撑点数为 $N_s=2N_h+2N_v$。对于 $41\times20$ 网格，$N_h=120$，$N_v=57$，总支持点数为354；其中不包含原始边界节点和四个角点。

三维采用相同的边界位移传播原则，但为了控制全局 IDW 成本，六个参数面分别在两个参数方向将原始区间三等分。保留所有新增的面内点以及每条原始单元边上的两个新增点，并删除原始面网格节点。一个 $N_u\times N_v$ 参数面的支撑点数为：

$$
N_{\mathrm{face}}=
[3(N_u-1)+1][3(N_v-1)+1]-N_uN_v.
$$

例如 $11\times11\times11$ 网格的每个面包含840个新增支撑点，六个面共5040个。程序对查询点和支撑点分块累计全局 IDW 分子、分母，以限制峰值显存。

### 9.3 边界位移与全局传播

在每个边界参数位置计算离散网格双线性点 $\mathbf p_i$ 和精确曲线点 $\mathbf b_i$：

$$
\mathbf d_i=\mathbf b_i-\mathbf p_i.
$$

对于内部点的双线性坐标 $\mathbf p$，采用物理空间距离平方倒数：

$$
w_i(\mathbf p)=\frac{1}{\|\mathbf p-\mathbf p_i\|_2^2+\varepsilon}.
$$

修正后的训练目标为：

$$
\mathbf x^*=\mathbf x_{\mathrm{bil}}+rac{\sum_{i=1}^{N_s}w_i\mathbf d_i}{\sum_{i=1}^{N_s}w_i}.
$$

四条边界的全部支持点共同参与每个内部点的修正。实现按每256个查询点分块计算，以限制显存峰值。

该方法显著提高曲边附近监督点的几何一致性，但没有严格的逐单元内部性保证。在 $t=0.6$、$41\times20$ 网格的20,000个边界层随机点测试中，99.585%的修正点位于精确物理域内。

## 10. 主坐标损失

随机点空间权重为：

$$
w(\boldsymbol{\xi})=1+16[(\xi-0.5)^4+(\eta-0.5)^4].
$$

主损失定义为：

$$
L_{\mathrm{coord}}=2\operatorname{mean}[(w(\boldsymbol{\xi})(\widehat{\mathbf x}-\mathbf x^*))^2].
$$

二维系数2将对两个坐标分量的均值转换为每点坐标平方距离的均值，即：

$$
L_{\mathrm{coord}}=\operatorname{mean}_{b,j}\left[w_j^2((\widehat{x}_{b,j}-x^*_{b,j})^2+(\widehat{y}_{b,j}-y^*_{b,j})^2)\right].
$$

代码中权重先乘误差再平方，因此实际贡献为 $w^2$。

## 11. Jacobian 合法性损失

当前默认使用有限差分计算硬约束后最终映射的 Jacobian：

$$
J=\begin{bmatrix}
\partial x/\partial\xi & \partial x/\partial\eta\\
\partial y/\partial\xi & \partial y/\partial\eta
\end{bmatrix}.
$$

内部位置使用中心差分，边界附近使用二阶单侧差分，保证全部查询点位于计算域内。损失为：

$$
L_J=\operatorname{mean}[\operatorname{ReLU}(J_{\min}-\det J)^2],
$$

其中 $J_{\min}=10^{-4}$，有限差分步长为 $h=10^{-3}$。只有低于阈值的 Jacobian 受到惩罚，梯度仍通过预测坐标传播至网络参数。

## 12. 法向损失

代码保留边界法向对齐损失：

$$
L_n=\operatorname{mean}[(1-\widehat{\mathbf v}_{\mathrm{in}}\cdot\widehat{\mathbf n}_{\mathrm{exact}})^2].
$$

当前 `boundary_normal_loss_enabled=false`，因此正式训练中 $L_n=0$，只在消融实验中使用。

## 13. 当前总损失

当前 `current` 映射、均匀随机采样、无法向损失、无残差正则时：

$$
\boxed{L=L_{\mathrm{coord}}+0.01L_J}.
$$

一般形式为：

$$
L=L_{\mathrm{coord}}+\lambda_JL_J+\lambda_nL_n+\lambda_rL_r.
$$

当前 $\lambda_J=0.01$，$\lambda_n=0$，$\lambda_r=0$，不存在边界坐标损失。

## 14. 优化与学习率

当前使用 Adam，参数如下：

| 参数 | 当前值 |
|---|---:|
| 初始学习率 | $4\times10^{-4}$ |
| 学习率保持轮数 | 1000 |
| 更新间隔 | 500 epoch |
| 衰减系数 | 0.8 |
| 停止继续衰减阈值 | $10^{-5}$ |
| 最大 epoch | 100000 |
| batch size | 64 |
| 梯度裁剪范数 | 1.0 |
| 随机种子 | 20260802 |

这里的 warmup 是前1000个 epoch 保持初始学习率，并非线性升温。之后每500个 epoch 执行：

$$
\mathrm{lr}_{k+1}=0.8\,\mathrm{lr}_k.
$$

## 15. 非有限值处理

每个 batch 依次检查损失、反向传播、裁剪梯度总范数、检查梯度有限性，再提交 Adam 更新。当前配置在损失或梯度出现 NaN/Inf 时跳过该 batch、清空梯度并输出警告。

## 16. 固定网格 MSE 与最佳模型

训练梯度来自随机点损失，但每个 epoch 还计算完整数据集严格内部网格节点的固定网格 MSE。边界节点不参与该指标。

当前设置：

```yaml
calculate_dataset_mse: true
dataset_mse_interval: 1
best_model_metric: dataset_mse
```

最佳模型因此选择为：

$$
e_{\mathrm{best}}=\arg\min_e L_{\mathrm{MSE}}^{(e)}.
$$

梯度优化目标与最佳 checkpoint 选择指标不同：前者为 $L_{\mathrm{coord}}+0.01L_J$，后者为固定内部节点数据集 MSE。

## 17. 提前停止条件

### 17.1 连续低训练损失

训练损失连续100个 epoch 满足 $L_{\mathrm{train}}<10^{-6}$ 时停止。MSE 阈值停止当前关闭。

### 17.2 稳定平台停止

当前平台指标为训练损失，窗口长度为200，连续要求为100。最近窗口计算 $q_{0.1}$、$q_{0.9}$ 和中位数 $m$，窗口宽度为：

$$
W=q_{0.9}-q_{0.1}.
$$

稳定条件为：

$$
W\le\operatorname{clip}(0.1|m|,10^{-6},10^{-5}).
$$

接近历史最佳窗口中位数要求：

$$
m\le m_{\mathrm{best}}+\max(10^{-7},0.5|m_{\mathrm{best}}|).
$$

长期改善由窗口下分位数判断：

$$
r_{\mathrm{improve}}=\frac{a-q_{0.1}}{\max(|a|,10^{-12})}.
$$

若 $r_{\mathrm{improve}}>0.005$，则认为发生显著改善并重置连续平台计数。窗口同时稳定、接近历史最佳且没有显著改善，连续100次后停止。

## 18. 每个 epoch 的完整流程

1. 随机打乱训练数据；
2. 组成64个几何样本的 batch；
3. 读取边界点、参数曲线和 PDE 网格；
4. 生成400个 batch 共享的均匀随机计算域点；
5. 对 PDE 网格执行双线性插值；
6. 将四条精确边界的每个网格区间四等分，只保留每段三个内部支撑点；
7. 使用全局边界位移 IDW 修正监督目标；
8. 为查询点生成 Jacobian 有限差分模板；
9. 批量前向计算中心点与差分点；
10. 从参数曲线直接计算四个精确投影点；
11. 执行边界硬约束融合；
12. 计算坐标损失与 Jacobian 合法性损失；
13. 检查损失、反向传播和梯度；
14. 裁剪梯度并执行 Adam 更新；
15. 按 batch 实际样本数统计 epoch 损失；
16. 计算完整数据集固定内部节点 MSE；
17. 更新最佳模型；
18. 判断损失阈值及平台收敛；
19. 更新学习率；
20. 写入日志与 TensorBoard。

## 19. 训练输出

每次训练保存：

```text
best.pt
best_model.pth
final.pt
final_model.pth
```

`*.pt` 保存模型状态、有效配置、网格规模、模型名称及 epoch；`*_model.pth` 只保存 `state_dict`。此外还保存实际配置、训练历史 CSV、训练损失图、固定网格 MSE 图、训练摘要 JSON、TensorBoard 记录、最佳与最终指标、训练时间、停止原因及学习率信息。

## 20. 可用于论文表述的核心贡献

1. **参数曲线驱动的边界硬约束**：将精确参数曲线直接嵌入网络输出，消除边界拟合误差和边界损失权重调节。
2. **全局曲边位移 IDW 监督目标**：利用精确边界与离散网格弦之间的位移修正随机点标签，提高曲边附近监督点的几何一致性。
3. **有限差分 Jacobian 合法性约束**：在避免计算坐标高阶自动微分开销的同时，抑制翻转和退化单元。

## 21. 论文中应明确披露的限制

1. 全局 IDW 提高了域内比例，但尚不能严格保证所有目标点位于对应曲边单元内部。
2. 空间权重实际以 $w^2$ 作用于平方误差，论文公式必须与代码保持一致。
3. 最佳模型按固定网格 MSE 选择，而平台停止默认使用随机点训练损失，两者目标不同。
4. `current` 硬约束保证边界坐标精确，但不能单独保证内部 Jacobian 为正，因此仍需 Jacobian 损失。
5. 正式结论应在多类边界和至少三个随机种子上报告均值与标准差。
6. 当前 HMONet 使用 PDE 网格标签监督训练，不应称为纯物理自监督方法；PI-HMONet 才属于 PDE 残差自监督模型。
