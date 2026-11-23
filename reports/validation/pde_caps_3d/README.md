# 九类三维PDE端面接入及二维既有结果核验

核验日期：2026-11-23。正式代码中两端面分别使用201×201节点的二维Winslow解，精确Coons边界加三次样条位移；四侧面保留原解析几何，在原单位矩形参数域使用TFI。没有参数重分布、没有上下端面求值复用。体内标签由三维Winslow求解器生成。

新数据集名称版本统一为1；geometry_revision=3、cap_parameterization=planar_winslow_coons_cubic_v1用于区分旧参数面。旧文件不自动改名、删除或覆盖；生成同名旧格式数据会拒绝静默复用，可明确使用--force重新生成。

## 三维端点检查

共35个训练外推端点/评估参数组合。每个端面检查201×201、401×401采样，随后检查数据集规模的六面与体PDE网格。下表的Scaled Jacobian为体网格最小值；只证明被测离散网格通过局部单元检查，不代表连续映射的全局双射证明，也不是完整500样本的逐一合法性认证。

| Boundary | Grid nodes | Parameters tested | Minimum scaled Jacobian | Invalid cells | NumPy/Torch max error |
|---|---|---|---|---|---|
| cah_t | 11×21×11 | 0.06, 0.08, 0.34, 0.38 | 0.0070095271 | 0 | 2.22e-16 |
| swb_a | 11×21×11 | 0.02, 0.04, 0.3, 0.34 | 0.781701 | 0 | 4.44e-16 |
| run_e | 61×21×11 | 0.55, 1.5, 1.52, 1.55 | 0.0077644919 | 0 | 1.78e-15 |
| teb_b | 11×21×11 | 0.08, 0.12, 0.32, 0.36 | 0.48170958 | 0 | 2.22e-16 |
| bad_a | 11×21×11 | 10.0, 25.0, 85.0, 105.0 | 0.93337186 | 0 | 4.44e-16 |
| lad_a | 11×21×11 | 0.0, 0.04, 0.16, 0.2 | 0.6319527 | 0 | 2.22e-16 |
| srt_r | 11×21×11 | 0.52, 0.55, 0.92, 0.96 | 0.1390717 | 0 | 2.22e-16 |
| sad_c | 11×21×11 | 0.5, 0.55, 1.35, 1.42 | 0.12815956 | 0 | 2.22e-16 |
| tbp_a | 11×21×11 | 0.0, 90.0, 100.0 | 0.19698054 | 0 | 2.22e-16 |

run_e的11×21×11在深凹端面出现非法四边形；21×21×11在1.50、1.52产生20个翻转体单元，在1.55端面也未通过。用户确认后正式默认改为61×21×11；其余八类仍11×21×11。范围不变。完整失败案例也保留在JSON，未用删除失败记录的方式呈现结果。

复核命令：

```powershell
python scripts/data/check_pde_cap_families.py --families all --split ext
python scripts/data/check_run_e_grids.py --grids 11x21x11 21x21x11 61x21x11 --values 0.55 1.50 1.52 1.55
```

## 三维实际流程测试

每类生成外推训练两端的2个样本；使用正式模型架构、400个训练查询点，HMONet、MeshONet、PI-MeshONet各训练1轮。逐类完成连续面读取、损失反向传播、模型保存、三个模型训练对比图和五方法评估，评估启用体网格与z_mid。测试产物与正式500样本结果隔离保存。1轮测试仅验证可运行和数值有限性，不证明正式训练收敛或预测质量。

| Boundary | Dataset read | HMONet | MeshONet | PI-MeshONet | Training comparison | Five-method evaluation |
|---|---|---|---|---|---|---|
| cah_t | Pass | Pass | Pass | Pass | Pass | Pass |
| swb_a | Pass | Pass | Pass | Pass | Pass | Pass |
| run_e | Pass | Pass | Pass | Pass | Pass | Pass |
| teb_b | Pass | Pass | Pass | Pass | Pass | Pass |
| bad_a | Pass | Pass | Pass | Pass | Pass | Pass |
| lad_a | Pass | Pass | Pass | Pass | Pass | Pass |
| srt_r | Pass | Pass | Pass | Pass | Pass | Pass |
| sad_c | Pass | Pass | Pass | Pass | Pass | Pass |
| tbp_a | Pass | Pass | Pass | Pass | Pass | Pass |

三维数值数据现在统一检查：PDE网格上的边界节点与boundary_samples.npy一致；读取连续曲面时，还会核对连续面与边界样本一致。后者用于HMONet；前者即使MeshONet/PI-MeshONet不读取几何对象也执行。

保存的端面信息包括原精确面定义、两个方向的节点向量、三次样条位移控制系数及求解诊断；恢复时不再求二维PDE。网络输入仍使用指定体网格规模的去重边界采样点，不使用201×201密集网格作为网络输入。

## 二维前六类现有结果

检查best/final权重是否存在且有限、训练停止原因、三模型训练对比图、配置中的101/201/401评估汇总、保存的五方法网格及评估所用模型。不能仅凭results.json存在就判断全部结束。

| Dataset | Three-model training | Evaluation cases | Completeness |
|---|---|---|---|
| air_t_ext_1_500_21_21 | 完成，best/final权重有限 | 6 | 存在下列待补项 |
| esa_ext_1_500_41_41 | 完成，best/final权重有限 | 6 | 存在下列待补项 |
| rse_ext_1_500_41_41 | 完成，best/final权重有限 | 15 | 完整 |
| run_ext_1_500_41_41 | 完成，best/final权重有限 | 6 | 完整 |
| scc_ext_1_500_41_41 | 完成，best/final权重有限 | 9 | 完整 |
| vsc_ext_1_500_41_41 | 完成，best/final权重有限 | 9 | 完整 |

- air_t_ext_1_500_21_21：401×401、0.65有汇总，但缺少generated_meshes.npz，未找到其余方法的完整网格缓存，不能视为全部可复用。
- esa_ext_1_500_41_41：401×401、0.72及0.75仍引用旧PI-MeshONet（20261118_1012），不是本机最新训练的最佳模型，应补评估。
- rse、run、scc、vsc：上述完整性检查通过。网格中某方法出现非法单元是质量结论，不等同于评估程序未完成；没有删除不利结果。
- 三个尚未完整训练的默认二维族是tri_p、gsp_a、fca_a；本机未发现这三个默认数据集的训练结果。最新正式命令文件包含先生成与后训练评估的步骤。

建议补评估（不重新训练）：

```powershell
python scripts/train/train_evaluate.py --dimension 2 --dataset air_t_ext_1_500_21_21 --models hmonet meshonet pimeshonet --evaluate-only --checkpoint best --grids 401 --values 0.65 --seed 20261030 --run-tag v1 --force
python scripts/train/train_evaluate.py --dimension 2 --dataset esa_ext_1_500_41_41 --models hmonet meshonet pimeshonet --evaluate-only --checkpoint best --grids 401 --values 0.72 0.75 --seed 20261030 --run-tag v1 --force
```

本次只核查既有二维结果，没有重跑这些401评估，也没有覆盖或删除旧模型/结果。

## 文档与可视化

- [二维完整生成命令](../../../data/2D/generation_commands.md)
- [三维完整生成命令](../../../data/3D/generation_commands.md)
- [二维正式实验命令](../../../configs/2D/formal_external_commands.md)
- [三维正式实验命令](../../../configs/3D/formal_external_commands.md)
- [十八类边界图和公式](../../../outputs/boundaries/README.md)

上述相对链接以项目目录为根；可视化目录按2D/3D及边界缩写组织。每类包括默认参数图、训练端点及评估参数对比图、公式和生成函数说明，三维另有逻辑z_mid截面。z_mid不是所有弯管的全局z常数平面。

## 生成与评估的一致性及回归检查

九类测试数据的上端参数，数据集 PDE 标签与评估器重新生成的 PDE 网格逐点差值均为 0，详见 `label_reference_checks.json`。该检查使用测试数据规模，不能替代全部 500 个样本的质量认证。

端面序列化与批量插值测试 17 项通过；九类端面、生成、投影等测试组修正两条旧版本/规模预期后通过。追加的批量命令、几何一致性、三维评估范围和生成合法性检查组 15 项通过。九类各三个模型的一轮训练用于验证读写、梯度、模型保存、训练对比与评估流程，不代表已完成正式训练或收敛验证。

最终追加：默认规模、数据读写和批量生成规则 10 项回归通过；训练快照及新旧参数面复用隔离 2 项通过。新训练记录加入 `dataset_geometry`（边界族、修订号、端面和侧面方案），同名旧数据训练的模型不会被误判为新参数面的已完成模型。

## 上传服务器前的再次核查

最新九类完整流程检查、边界一致性检查、旧模型隔离修复及 run_e 例外，见 [服务器运行前核查](server_preflight.md)。

后续默认规模已调整：其余八类21×21×11，run_e为41×41×11；默认评估41×41×11、81×81×11。新规模核查、81规模完整评估计时与run_e凹六面体局部图见 [网格规模复核](grid_scale_review.md)。此前11×21×11的流程记录属于当时配置，保留作为历史验证记录。

## 2026-11-23 默认规模后续调整

run_e默认规模按要求改为41×41×11。实测槽深0.55、1.50的端面及体网格通过；评估参数1.52、1.55的PDE体网格分别存在16、20个翻转单元，1.55的两个端面各有2个内凹四边形。数据集生成会拦截非法端面；评估会正常记录网格质量，不能把求解收敛或评估结束当作网格合法。61×21×11的通过记录保留在核验报告中。 上文61×21×11结果为当时实测记录，不应改写为41×41×11的结果；新测试见 `run_e_41x41x11.json` 及 `run_e_41x41_face_diagnostics.json`。
