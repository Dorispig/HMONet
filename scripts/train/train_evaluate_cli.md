# 训练与评估串联命令行说明

`train_evaluate.py` 会先完成所选模型和数据集的训练，再使用本次保存的模型进行评估。

运行前：

```powershell
conda activate MGNet
cd E:\projects\HMONet
```

## 一、完整命令示例

以下为 PowerShell 多行命令。每行末尾的反引号表示续行，后面不能有空格。

```powershell
python scripts/train/train_evaluate.py `
  --dimension 2 `
  --train-config configs/2D/train.yaml `
  --evaluate-config configs/2D/evaluate.yaml `
  --checkpoint best `
  --train-args "--model all --dataset-mode both --jacobian-method finite_difference --jacobian-fd-step 0.001 --num-epochs 100000 --batch-size 64 --lr 0.0004 --seed 20260802 --device cuda --hmonet-mapping-mode current" `
  --evaluate-args "--model both --inference-warmup-runs 3 --inference-timing-repeats 10 --compute-grid-quality --quality-metrics both"
```

该示例假定训练 YAML 已填写 air_t 的两个 **21×20** 数据集路径：

```yaml
datasets:
  interpolation: data/raw/2D/air_t_int_0_1000_21_20
  extrapolation: data/raw/2D/air_t_ext_0_1000_21_20
```

命令将分别训练两个数据集上的 MeshONet、HMONet、PI-MeshONet、PI-HMONet、Attention-PI-HMONet，共10次训练；随后仅比较 TFI、PDE、MeshONet、HMONet，网络使用本次训练的最佳模型。

### 逐行说明

| 命令行 | 修改的内容 |
|---|---|
| `python scripts/train/train_evaluate.py` | 启动训练—评估串联入口。 |
| `--dimension 2` | 选择二维流程；设为 `3` 使用三维流程。 |
| `--train-config ...` | 指定训练 YAML 文件。 |
| `--evaluate-config ...` | 指定评估 YAML 文件。 |
| `--checkpoint best` | 评估本次训练的最佳模型；设为 `final` 使用最终模型。 |
| `--train-args "..."` | 将引号内的参数传递给训练入口，覆盖训练 YAML。 |
| `--evaluate-args "..."` | 将引号内的参数传递给评估入口，覆盖评估 YAML。 |

### --train-args 中各参数的含义

| 参数 | 修改的内容 |
|---|---|
| `--model all` | 五个模型分别训练；`both` 仅训练 HMONet、MeshONet，`pi` 训练三个 PI 模型，也可单选 `attentionpihmonet`。 |
| `--dataset-mode both` | 内插、外推数据集分别训练；也可以选择 `interpolation` 或 `extrapolation`。 |
| `--nx 21` | 可选的一致性检查；训练规模自动读取数据集，显式值不一致时中止。 |
| `--ny 20` | 可选的一致性检查；训练规模自动读取数据集，显式值不一致时中止。 |
| `--num-epochs 100000` | 最大训练轮数；启用提前停止时可能提前结束。 |
| `--batch-size 64` | 每个训练批次的样本数量。 |
| `--lr 0.0004` | 初始学习率。 |
| `--seed 20260802` | 控制初始化、数据遍历及随机采样的随机种子。 |
| `--device cuda` | 使用 CUDA 训练；也可以选择 `cpu` 或 `auto`。 |
| `--jacobian-method finite_difference` | HMONet 的 Jacobian 损失采用有限差分；可改为 `autograd`。仅在该损失开启时生效。 |
| `--jacobian-fd-step 0.001` | Jacobian 有限差分的计算域步长。 |
| `--hmonet-mapping-mode current` | HMONet 内部映射方式；也可以选择 `coons_residual`。 |

训练损失、收敛条件、学习率调度等未在示例中覆盖的参数，继续使用训练 YAML 的设置。其他训练命令行参数也可以放入 `--train-args`，完整列表可查看：

```powershell
python scripts/train/train_2d.py --help
```

### --evaluate-args 中各参数的含义

| 参数 | 修改的内容 |
|---|---|
| `--model both` | 仅评估 TFI、PDE、MeshONet、HMONet，不影响训练模型选择。模型选项详见文末。 |
| `--inference-warmup-runs 3` | 网络计时前预热3次。 |
| `--inference-timing-repeats 10` | 网络计时运行10次，按评估代码的规则统计时间。 |
| `--compute-grid-quality` | 开启网格质量计算。 |
| `--quality-metrics both` | 计算当前实现的所有质量指标，但仍受各指标开关控制。 |

评估参数值、评估网格规模以及其他未覆盖的选项，继续使用评估 YAML 的设置。其他允许的评估参数可以放入 `--evaluate-args`，完整列表可查看：

```powershell
python scripts/evaluate/evaluate_2d.py --help
```

## 二、串联入口支持的全部参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dimension` | `2` | 维度，可选 `2`、`3`。 |
| `--train-config` | `configs/<维度>D/train.yaml` | 训练配置文件。 |
| `--evaluate-config` | `configs/<维度>D/evaluate.yaml` | 评估配置文件。 |
| `--checkpoint` | `best` | 选择本次训练的 `best.pt` 或 `final.pt`。 |
| `--train-args` | 空 | 训练命令行覆盖参数，整个字符串需要加引号。 |
| `--evaluate-args` | 空 | 评估命令行覆盖参数，整个字符串需要加引号。 |
| `--evaluate-methods` | 未指定 | 可选的评估方法列表，例如 `tfi,pde,meshonet,hmonet`；不能与 `--evaluate-args` 中的 `--model` 同时使用。 |
| `--check-only` | 关闭 | 只检查数据读取、训练网格和评估 case，不进行训练或评估。 |
| `-h` / `--help` | — | 显示入口帮助后退出。 |

`--check-only` 和 `--help` 属于检查、帮助模式，因此不放在正式训练命令中。

## 三、运行前检查

在完整命令后增加 `--check-only`，即可只进行预检查。例如：

```powershell
python scripts/train/train_evaluate.py `
  --dimension 2 `
  --train-config configs/2D/train.yaml `
  --evaluate-config configs/2D/evaluate.yaml `
  --train-args "--model all --dataset-mode both" `
  --evaluate-args "--model both" `
  --check-only
```

检查通过后，删除 `--check-only` 即可开始运行。

注意：该检查不会实际运行模型，不代表已经验证显存容量、完整训练或完整评估。

## 四、指定单个数据集

```powershell
python scripts/train/train_evaluate.py `
  --dimension 2 `
  --train-args "--dataset data/raw/2D/esa_int_0_1000_20_20 --dataset-mode interpolation --model both"
```

使用 `--dataset` 后，只训练指定的数据集，不再从 YAML 选择两套数据。

评估 YAML 的 `cases` 中必须存在与数据集文件夹名称一致的条目，例如：

```yaml
cases:
  esa_int_0_1000_20_20:
    # 对应的评估配置
```

单独训练并评估 Attention-PI-HMONet：

```powershell
python scripts/train/train_evaluate.py `
  --dimension 2 `
  --train-args "--model attentionpihmonet --dataset-mode interpolation" `
  --evaluate-methods tfi,pde,attentionpihmonet
```

## 五、参数传递限制

以下参数由串联脚本自动管理，**不能放进 --evaluate-args**：

```text
--config
--case
--all-cases
--dataset
--methods
--checkpoint
--hmonet-checkpoint
--meshonet-checkpoint
--pimeshonet-checkpoint
--pihmonet-checkpoint
--attentionpihmonet-checkpoint
--hmonet-mapping-mode
--pihmonet-mapping-mode
--attentionpihmonet-mapping-mode
```

其中：

- 评估 case 根据本次训练的数据集名称确定。
- checkpoint 使用本次训练保存的文件。
- HMONet、PI-HMONet、Attention-PI-HMONet 内部映射方式与本次训练保持一致。
- 默认比较 TFI、PDE，以及本次实际训练的模型；可通过 `--evaluate-args "--model both"` 等选择评估模型，或通过外层 `--evaluate-methods` 选择方法。两种选择方式不能同时使用。
- 更换评估配置文件请使用外层 `--evaluate-config`。
- 更换训练配置文件请使用外层 `--train-config`，不能在 `--train-args` 内填写 `--config`。

## 六、路径和引号

配置文件及数据集的相对路径按项目根目录解析。

在 `--train-args`、`--evaluate-args` 内，路径建议使用正斜杠：

```text
E:/projects/HMONet/data/raw/2D/esa_int_0_1000_20_20
```

如果只传递一个不带值的开关，建议使用等号写法：

```powershell
python scripts/train/train_evaluate.py --evaluate-args="--no-compute-grid-quality"
```

## 七、执行顺序和输出

执行顺序为：

1. 检查训练数据、网格规模及对应评估 case。
2. 分别完成所有选定模型、数据集的训练。
3. 收集本次训练保存的 checkpoint。
4. 按数据集分别评估本次模型。

训练失败后不会继续进入评估阶段。

输出位置沿用原有配置与目录结构：

```text
outputs/
├── training/       # 训练日志、摘要、曲线等
├── checkpoints/    # 模型 checkpoint 和参数文件
├── tensorboard/    # TensorBoard 日志镜像
└── results/        # 评估网格、质量、对比图及统计
```

该入口不会自动启动 TensorBoard 服务。

三维使用 `--dimension 3`，默认读取 `configs/3D/train.yaml` 和 `configs/3D/evaluate.yaml`；训练网格自动读取数据集；`--nx/--ny/--nz` 仅用于可选的一致性检查。底层命令行帮助为：

```powershell
python scripts/train/train_3d.py --help
python scripts/evaluate/evaluate_3d.py --help
```

## 八、每次训练的参数归档

二维训练在第一轮开始前，将包含命令行覆盖值的实际生效配置保存到：

```text
configs/2D/<dataset_name>/HMONet/<current或coons_residual>/<YYYYMMDD_HHMM>/train.yaml
configs/2D/<dataset_name>/MeshONet/<YYYYMMDD_HHMM>/train.yaml
```

每份文件仅指向本次训练的数据集，模型也固定为本次模型。归档时间与 checkpoint 一致；同分钟重复训练添加 `_02` 等后缀，不覆盖旧配置。可用 `--train-config` 指定归档 YAML 来重训。主配置 `configs/2D/train.yaml` 不会被改写。

## 九、PI-MeshONet 自监督训练与评估

```powershell
python scripts/train/train_evaluate.py --dimension 2 --train-args "--model pimeshonet --dataset-mode both --pi-pde-loss-weight 0.0001 --pimeshonet-boundary-loss-weight 1.0 --pi-interior-sample-count 128"
```

使用 `pimeshonet_training` 配置，只优化 PDE 残差和边界误差。模型结构使用 `model.common`，可在 `model.pimeshonet` 中覆盖结构参数。固定网格标签仅供 MSE 监测；最佳模型选择和平台判断固定使用训练损失，标签相关停止条件关闭。`both` 仍仅训练 HMONet 和 MeshONet，`all` 训练四个模型。评估对比顺序为 TFI、PDE、MeshONet、PI-MeshONet、HMONet、PI-HMONet（仅显示选中的方法）。

差分步长可以通过 `--pi-fd-step 0.05`（各轴相同）或二维 `--pi-fd-step 0.05 0.04`、三维 `--pi-fd-step 0.05 0.04 0.03` 指定。默认按训练网格每个方向取 `1/(n-1)`；步长必须位于 `(0,0.5)`，默认步长要求每轴至少4个节点。采样中心限制在每轴 `[h,1-h]`，避免差分模板越出计算域。二维9点、三维19点，均在一个批次前向中计算，无输入坐标自动微分；训练参数仍正常反向传播。

独立评估：

```powershell
python scripts/evaluate/evaluate_2d.py --case esa_int_0_1000_20_20 --methods tfi,pde,meshonet,pimeshonet,hmonet
python scripts/evaluate/evaluate_2d.py --case esa_int_0_1000_20_20 --model pimeshonet --pimeshonet-checkpoint outputs/checkpoints/2D/esa_int_0_1000_20_20/PI-MeshONet/<时间>/best.pt
```

`--pimeshonet-boundary-projection none|direct|inverse_distance` 控制 PI-MeshONet 的边界后处理（默认 none，原始输出直接参与比较）。维度为3时选择 `--dimension 3`；独立入口用 `train_3d.py` 和 `evaluate_3d.py`。各数据路径和网格规模必须匹配对应 YAML。

## 十、简化平台收敛条件

主配置使用 `training.plateau_enabled` 总开关。窗口满后，比较当前窗口中位数与历史最佳窗口中位数，稳健波动要求 `Q90-Q10 <= clip(相对容差 * abs(中位数), 绝对下限, 绝对上限)`。
“无显著改善”和“短窗口连续稳定”分别累计：无显著改善计数只在出现显著改善时归零，短窗口稳定计数只在短窗口不稳定时归零。达到无显著改善耐心且当前窗口接近历史最佳后，长期波动稳定或短窗口连续稳定任一分支通过即可停止；停止不会早于 `convergence_start_epoch`。失败平台仍独立判定。
最佳模型保存规则不变。连续训练损失低于阈值仍是独立的停止路径，不必等待平台条件。

```powershell
python scripts/train/train_evaluate.py --train-args "--plateau-enabled --convergence-start-epoch 3000 --plateau-window 200 --success-plateau-patience 1000 --failure-plateau-metric-ratio 1.5"
python scripts/train/train_evaluate.py --train-args "--no-plateau-enabled"
```

新规则不使用旧的独立开始轮数、连续稳定次数及多个平台开关。未包含 `plateau_enabled` 的历史配置保留旧逻辑；可通过命令行 `--plateau-enabled` 切换到新逻辑。

## PI-HMONet 与四模型选择

新增 `pihmonet`，`both` 保持 HMONet＋MeshONet，`all` 现在选择四种模型。
PI-HMONet 训练使用 `--pihmonet-mapping-mode current/coons_residual`；评估还可设为 `both`。
PI-HMONet 专属覆盖包括 pihmonet-pde-loss-weight、
pihmonet-interior-sample-count、pihmonet-fd-step、
pihmonet-jacobian-loss-weight 和 pihmonet-jacobian-minimum。
PI-MeshONet 同样支持 `--pimeshonet-*` 前缀的前四项；专属覆盖优先于原有 `--pi-*` 公共覆盖。
评估支持 `--pihmonet-checkpoint`、`--pihmonet-boundary-projection` 与 `--methods tfi,pde,pihmonet`。
训练评估串联时，训练模型和映射由 `--train-args` 指定；评估可以另选本次训练模型的子集，检查点和内部映射仍自动使用本次训练的设置。
完整流程、公式及二维三维命令示例见项目中的 `scripts/train/pi_models.md`。


## Jacobian 损失求导方式（二维、三维）

HMONet 的 `jacobian_loss_enabled` 开关和权重含义不变；默认采用有限差分。
`--jacobian-method finite_difference` 使用有限差分，`--jacobian-method autograd` 恢复原自动微分。
`--jacobian-fd-step 0.001` 设置计算域步长，仅在有限差分模式生效。
内部点使用中心差分，边界附近使用二阶单侧差分，查询点不越出计算域。
二维将中心与四个邻点、三维将中心与六个邻点合并前向；主坐标损失只使用中心预测。
自动微分模式仍对完整硬约束输出求导。两种方式均允许损失反向传播到网络参数。
PI-HMONet 当前仍仅优化 PDE 损失，不因这个默认值改变而自动增加 Jacobian 损失。

```powershell
python scripts/train/train_2d.py --model hmonet --jacobian-loss-enabled --jacobian-method finite_difference --jacobian-fd-step 0.001
python scripts/train/train_3d.py --model hmonet --jacobian-loss-enabled --jacobian-method autograd
python scripts/train/train_evaluate.py --dimension 2 --train-args "--model all --dataset-mode both --jacobian-method finite_difference --jacobian-fd-step 0.001"
```

损失仍为 `mean(relu(jacobian_minimum - det(J))**2)`；差分步长会影响近似精度，不能保证与自动微分在尖点或分段接头处完全一致。


## 训练全部模型，仅评估指定方法

可以在 `--evaluate-args` 中设置 `--model`，无需另写 `--evaluate-methods`。
以下命令按训练 YAML 的其余参数，在 air_t 的内插、外推数据集上分别训练四个模型，然后仅评估 TFI、PDE、MeshONet、HMONet：

```powershell
python scripts/train/train_evaluate.py --dimension 2 --train-config configs/2D/train.yaml --evaluate-config configs/2D/evaluate.yaml --checkpoint best --train-args "--model all --dataset-mode both --jacobian-method finite_difference --jacobian-fd-step 0.001" --evaluate-args "--model both"
```

| 设置位置和参数 | 含义 |
|---|---|
| `--train-args "--model all ..."` | 训练全部四个网络。 |
| `--train-args "... --dataset-mode both"` | 内插、外推分别训练，共8次训练。数据集路径由训练 YAML 指定。 |
| `--evaluate-args "--model both"` | 评估 TFI、PDE、MeshONet、HMONet，不评估两个 PI 模型。 |
| `--evaluate-args "--model all"` | 评估 TFI、PDE 和全部四个网络，共六种方法。 |
| `--evaluate-args "--model hmonet"` | 评估 TFI、PDE、HMONet；也可单选 `meshonet`、`pimeshonet`、`pihmonet`。 |
| `--checkpoint best` | 评估本次训练的最佳检查点；改为 `final` 使用最终检查点。 |

不指定评估模型或方法时，默认评估 TFI、PDE 和本次训练的全部网络。所选网络必须参与本次训练，评估不会回退查找历史模型。评估 case 自动对应本次训练的数据集。

若需要任意方法组合（例如只比较 TFI、PDE），使用外层 `--evaluate-methods`：

```powershell
python scripts/train/train_evaluate.py --dimension 2 --train-config configs/2D/train.yaml --evaluate-config configs/2D/evaluate.yaml --checkpoint best --train-args "--model all --dataset-mode both --jacobian-method finite_difference --jacobian-fd-step 0.001" --evaluate-methods tfi,pde,meshonet,hmonet
```

该命令与前面的四方法评估效果一致。不要同时使用 `--evaluate-methods` 和 `--evaluate-args` 中的 `--model`，同时指定会报错。`--methods` 不能放入 `--evaluate-args`。

上述命令末尾追加 `--check-only` 可先检查数据集、网格、评估 case 及模型选择，不执行训练或评估。三维同样支持上述模型选择规则，改用 `--dimension 3` 和对应的三维配置文件即可。

### PI-HMONet Jacobian 损失

PI-HMONet 默认启用有限差分 Jacobian 合法性损失，并复用 PDE 模板点，不增加额外网络前向：

L = L_PDE + 0.01 * L_Jacobian

可使用 --pihmonet-jacobian-loss-weight 0.01 修改权重，使用
--pihmonet-jacobian-minimum 1e-4 修改最小目标值，或使用
--no-pihmonet-jacobian-loss-enabled 临时关闭。

## PI 两模型联合训练与评估

模型选择参数 --model pi 表示依次训练 PI-MeshONet 和 PI-HMONet。
在 train_evaluate.py 中使用时，训练完成后自动使用本次训练产生的两个
checkpoint，按 TFI、PDE、PI-MeshONet、PI-HMONet 的顺序评估：

    python scripts/train/train_evaluate.py --dimension 2 --train-args "--model pi --dataset-mode both" --evaluate-args "--model pi"

--model both 仍表示 HMONet 与 MeshONet；--model all 表示全部四个网络。

## 只重训 PI-HMONet 并与已有 PI-MeshONet 对比

训练参数使用 --model pihmonet，只训练 PI-HMONet。评估方法显式指定
tfi,pde,pimeshonet,pihmonet。本次 PI-HMONet 使用刚生成的 checkpoint；
PI-MeshONet 没有在本次重训，因此评估器自动读取对应数据集已有的最新兼容
checkpoint。若没有可用的 PI-MeshONet checkpoint，评估会明确报错。

## 四模型统一的常用训练参数

四个模型统一支持 --epochs、--batch-size、--seed、--device、--lr、
--lr-warmup-epochs、--lr-update-interval、--lr-decay-gamma、
--lr-stop-below、收敛参数以及 --interior-sample-count。

PI-HMONet 也支持与 HMONet 相同的 --jacobian-loss-enabled、
--jacobian-loss-weight 和 --jacobian-minimum。带模型名前缀的 PI 专属参数
仍然有效并具有更高优先级，例如 --pihmonet-jacobian-loss-weight。

不指定 --dataset 时，从 train.yaml 的 datasets.interpolation 和
datasets.extrapolation 读取默认数据集；--dataset-mode both 依次运行两者。

