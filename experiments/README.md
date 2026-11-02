# 消融实验套件

推荐使用精简入口：

```bash
python scripts/experiments/run.py ex9 e9_1 --grids 101 201 401
python scripts/experiments/run.py ex9 baseline all --tag physics_v1
```

第一个位置参数选择实验套件，后续位置参数选择一个或多个实验；数据集、随机种子和评估配置未指定时读取套件配置。原有
`run_ablations_2d.py` 长命令仍兼容。单变体实验不再生成重复的
`<实验名>/<实验名>/` 训练层级。已有结果不做破坏性搬移，运行器会递归发现并按指纹复用；
`results_catalog.json` 给出所有历史清单的位置。

所有 `ex1`--`ex10`（包括 `ex8_1`）均复用同一个动态 GPU 队列实现。队列中的最小任务固定为“一个实验变体、一个数据集、一个随机种子”的完整训练与评估；只有显式指定 `--train-only` 或 `--evaluate-only` 时才执行单阶段。`Ctrl+C`/`SIGTERM` 会中断整个队列、终止所有当前子任务并取消待运行任务，`--continue-on-error` 仅适用于普通任务异常。推荐统一使用 `python scripts/experiments/run.py exN ... --gpus ...`；旧的 `run_ablations_2d.py` 和 `run_baseline_2d.py` 入口也共享相同行为。

- `ex1/`：迁移前的全部历史消融实验、模型和评估结果，配置与指纹保持原样。
- `ex2/`：保留主干几何系数与自注意力的历史结构实验；Softplus 激活实验已删除。当前正式HMONet固定使用Branch×Trunk调制和Sigmoid激活。
- `ex3/`：Jacobian 损失与曲边目标插值的 2×2 组合消融。
- `ex4/`：约17.8万参数HMONet下的Jacobian损失与曲边目标插值组合消融。
- `ex5/`：沿用ex3基线训练策略的HMONet参数量消融。
- `ex6/`：沿用ex3基线参数的数据集大小消融，训练子集不落盘。
- `ex7/`：精简HMONet在曲边修正下的Jacobian与非归一化PDE损失组合消融。
- `ex8/`：Current硬约束下的多阶段监督/PDE训练策略消融。
- `ex8_1/`：复用ex8冻结基线，将ex8的五种训练策略统一改为Coons残差硬约束。
- `ex9/`：监督、PDE、Jacobian、Winslow、奇异值正则、混合采样和三种硬约束形式的物理训练实验；当前包含 `e1`--`e8`、`e9`、`e9_1`--`e9_4`。
- `ex10/`：以 `ex9/e9_3` 配置为基线，对比三种采样组成、PI-MeshONet 与 MeshONet，并支持两个500样本外推数据集。

运行时用 `--experiment-set exN` 或编号子套件（如 `--experiment-set ex8_1`）选择实验。通常每套实验独立读取自己的基线与评估配置；`ex8_1` 在实验定义中显式引用只读的 `ex8/baseline`，不会复制、训练或修改基线。各变体的训练、评估及汇总仍写入所选实验集目录。

## 评估输出

消融评估先使用套件内唯一的 `.eval_work/` 临时目录，完成后立即整理到：

```text
experiments/<exN>/eval/<数据集>/seed_<随机种子>/<run-tag>/<实验项>/<网格规模>/<参数值>/
```

临时目录不再创建随机哈希子目录。正式目录不再使用哈希缩写；每个实验集统一按“数据集/seed/run-tag/实验项”组织评估，评估指纹仅保存在 JSON 元数据中。写入前后仍会检查 Windows 经典路径上限的 90%。

每个“参数×网格规模×随机种子”的目录保留 `results.json` 和所有评估图片。该汇总文件包含方法耗时、边界误差、网格合法性以及各质量指标的最差值和平均值，足以完成后续汇总与指纹复用。逐单元 CSV/NPZ 和生成网格缓存不再长期保存。

跨随机种子的正式汇总保存在 `aggregate/<标签>/<网格规模>/`：

- `per_case_metrics.csv`：每个参数和随机种子的汇总指标；
- `mean_variance_by_case.csv`：跨随机种子的均值、方差和标准差；
- `missing_runs.csv`：缺失训练或评估记录；
- `evaluation_index.json`：各网格规模的汇总索引。

纯网格图片、网格质量图片、方法对比图片，以及训练曲线、检查点和 TensorBoard 文件均予以保留。评估图片默认生成；运行消融实验时可用 `--no-evaluation-plots` 关闭，也可用默认的 `--evaluation-plots` 显式开启。

只评估指定网格规模时使用 `--evaluate-only --evaluation-grid-shapes 101x101 201x201`；只补指定几何参数时再加 `--evaluation-values 0.04`。未指定 `--force` 时，只要评估指纹一致且对应 `results.json` 存在，就直接复用已有汇总结果，不会为生成已删除的逐单元文件或图片而重新评估。需要重算指标时才使用 `--force`。

每套实验的公共基线只作为只读参照，训练位于 `baseline/<数据集>/...`，评估位于 `eval/<数据集>/seed_<随机种子>/<run-tag>/baseline/...`。运行任何具体消融实验时，只训练和评估命令中选中的实验变体，绝不自动训练、强制重训或补评估基线，也不在具体实验目录中生成 `baseline`/`base` 副本。

每次消融运行都会只读检查“当前冻结基线配置”与“已经训练好的基线”指纹。若不一致，当前实验的全部数据集、随机种子和变体仍会正常完成训练及其独立评估；随后跳过依赖基线的对比汇总，在运行末尾抛出明确的基线指纹错误。基线只会在用户显式调用 `run_baseline_2d.py`，或在 `--ablations` 中明确写出 `baseline` 时运行。

如需在一条命令中先准备基线再运行实验，可显式把 `baseline` 放入选择列表。`--ablations baseline <实验名>` 会先运行一次基线，再运行指定实验；`--ablations baseline all` 会先运行一次基线，再运行本套全部实验。基线训练、检查点、清单和评估结果始终只保存在套件的 `baseline/` 中，不会进入具体实验目录。单独使用 `--ablations all` 仍只运行实验，不包含基线。

## 当前统一命令

推荐只使用短入口：

```bash
python scripts/experiments/run.py <suite> [baseline] <experiment|all> [options]
```

| 参数 | 含义 |
| --- | --- |
| `--dataset NAME` | 覆盖数据集；可重复传入多个数据集 |
| `--seed S...` | 一个或多个随机种子；`--seeds` 同义 |
| `--grids N...` | 二维方形评估节点数，支持 `101` 或 `101x101` |
| `--values V...` | 覆盖评估物理参数 |
| `--tag TAG` | 输出标签；不参与训练指纹，但区分目录 |
| `--device cuda:0` | 串行设备；与 `--gpus` 互斥 |
| `--gpus 0 1 2 3` | 动态多 GPU 任务队列 |
| `--queue-terminal-interval N` | 队列每 N 轮向主终端转发完整汇总 |
| `--train-only` | 只训练 |
| `--evaluate-only` | 只评估已有兼容训练 |
| `--force` | 替换所选运行；建议先 `--dry-run` |
| `--continue-on-error` | 普通任务失败后继续，最后仍报告失败 |
| `--plots/--no-plots` | 开启或关闭新评估图片，默认开启 |

各套实验的当前实验名和可复制命令见其目录内 README；完整工作流见[命令速查表](../docs/command_reference.md)与[复现规范](../docs/reproducibility.md)。
