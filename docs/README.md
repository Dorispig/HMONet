# HMONet 文档

从“无数据集、无训练结果”的完整复现流程开始，请先读根目录 [README](../README.md#从零复现九类外推数据集与正式实验)。本目录只保留与该流程配套的八个入口：

1. [环境、跨平台运行与代码同步](environment.md)
2. [命令与多 GPU 队列速查](command_reference.md)
3. [参数化数据集、边界族与曲线接口](dataset_generation.md)
4. [评估体系、指标与输出结构](evaluation_system.md)
5. [实验复现、指纹与 checkpoint 兼容性](reproducibility.md)
6. [已知限制与结果解释](known_limitations.md)
7. [二维 Poisson 下游案例](downstream_poisson_2d.md)
8. `README.md`（本索引）

二维、三维的逐项正式命令分别见 [configs/2D](../configs/2D/) 和 [configs/3D](../configs/3D/)。模型、损失、采样器、数值方法和评估代码的贡献规则见根目录 [CONTRIBUTING.md](../CONTRIBUTING.md)；HMONet 公式和训练架构见 [HMONET_TRAINING_ARCHITECTURE.md](../HMONET_TRAINING_ARCHITECTURE.md)。自动生成的消融报告不属于使用文档，统一保存在 `reports/ablations/`。

