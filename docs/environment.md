# 环境、跨平台运行与代码同步

## 安装

推荐 Python 3.10。正式训练和大规模评估建议使用 NVIDIA GPU；PyTorch/CUDA 版本必须与服务器驱动兼容。

```bash
conda create -n HMONet python=3.10 -y
conda activate HMONet
pip install -r requirements.txt
```

开发和完整测试额外安装：

```bash
pip install -r requirements-dev.txt
```

Linux 服务器若需要特定 CUDA wheel，应先从 PyTorch 官方渠道安装与驱动匹配的 `torch`/`torchvision`，再安装其余依赖，并确认固定版本没有被意外替换。

## 环境验证

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -m unittest discover -s tests -p "test_*.py" -v
python -m pytest tests -q
```

项目同时包含 `unittest` 类和 pytest 风格测试函数，因此发布前必须执行两条测试命令。只安装 `requirements.txt` 时可以运行项目，但不能声称已完成全部测试。

## Windows 与 Linux

所有命令均从项目根目录执行。PowerShell 和 Bash 的单行参数相同；多行命令分别使用反引号和反斜杠。服务器无桌面环境时绘图使用 Agg 后端并直接保存文件。

- Linux 路径和数据集名称区分大小写；
- 项目内路径使用 `/` 或 `pathlib`，不要新增机器相关的绝对路径；
- 多 GPU 子进程通过 `CUDA_VISIBLE_DEVICES` 隔离；
- `data/raw/`、`outputs/` 和实验运行结果不由 Git 自动传输；
- 跨机器复制数据和 checkpoint 时保持项目内相对路径不变。

旧记录中的 `E:\\projects\\HMONet_p\\...` 可用以下工具转换为当前相对路径：

```bash
python tools/migrations/migrate_portable_paths.py
```

该命令只改写 JSON/YAML 路径字符串，不修改模型参数、训练指纹或 checkpoint；重复运行是幂等的。

## 服务器代码更新

服务器存在本地修改时，先提交或建立备份分支，再同步远端；不要直接覆盖未备份工作。确认备份后，可从默认分支更新：

```bash
git fetch origin
git switch main
git pull --ff-only origin main
```

数据集、checkpoint、TensorBoard 和评估结果通常被 `.gitignore` 排除，不会随 `git pull` 自动更新。实验复用由训练指纹和文件完整性决定，与绝对路径及 run tag 无关；正式运行前建议先使用 `--dry-run` 检查任务计划。

