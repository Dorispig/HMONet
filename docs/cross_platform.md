# Windows 与 Linux 跨平台运行

项目路径均由代码文件位置推导，不依赖盘符或固定安装目录。配置、训练摘要和消融实验清单中的项目内路径统一保存为使用 `/` 的相对路径。

## 环境准备

建议在两个平台都创建同名环境，然后安装依赖：

```bash
conda create -n MGNet python=3.11 -y
conda activate MGNet
pip install -r requirements.txt
```

GPU 服务器应先根据服务器 CUDA 和驱动版本安装匹配的 PyTorch，再安装其余依赖。CPU 也可以运行，但训练和大规模评估会很慢。

## 启动方式

所有命令均从项目根目录执行。Windows PowerShell 和 Linux Bash 都可直接使用单行命令，例如：

```bash
python scripts/train/train_2d.py --config configs/2D/train.yaml
python scripts/evaluate/evaluate_2d.py --config configs/2D/evaluate.yaml
python scripts/experiments/run_baseline_2d.py --dataset air_t_int_0_1000_41_20 --seeds 20260802 20260917 20261031 --run-tag paper_v1 --dry-run
```

需要换行时，PowerShell 使用反引号，Bash 使用反斜杠；参数本身不变。

服务器无桌面环境时，绘图模块自动使用 Agg 后端，结果直接保存到文件。

## 搬运数据和训练结果

`data/raw/`、`outputs/` 以及 `experiments/` 下的运行结果默认不进入 Git。若要在 Linux 上复用 Windows 已训练模型，需要单独复制这些目录并保持其项目内相对位置。

旧记录中的 `E:\\projects\\HMONet_p\\...` 会在读取时自动重定位到当前项目根目录。也可以在搬运前或搬运后执行一次：

```bash
python tools/migrations/migrate_portable_paths.py
```

该命令只改写 JSON/YAML 中的路径字符串，不修改 checkpoint、训练指纹或模型参数。再次运行是幂等的。

## 消融实验复用规则

是否重新训练由训练参数指纹决定，与项目绝对路径和 `run-tag` 无关。参数指纹相同且训练摘要、checkpoint 完整时，脚本直接复用；只有训练参数发生变化、必要文件缺失或显式指定强制训练时才会重新训练。

建议先执行 `--dry-run` 检查计划。输出包含 `reuse-training` 时不会启动训练。

## Git 中保留的文件

消融实验的公共配置会被保留：

- `experiments/baseline/experiment.yaml`
- `experiments/baseline/evaluate.yaml`
- `experiments/<ablation>/experiment.yaml`
- `experiments/README.md`

checkpoint、TensorBoard、评估结果、聚合表格和临时清单仍被忽略。
