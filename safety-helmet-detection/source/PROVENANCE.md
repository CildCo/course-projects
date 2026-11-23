# 来源与原创说明

- 思路来源：工地/厂区安全帽佩戴检查的重复劳动；参考了公开的 hard-hat 检测产品和数据集组织方式。
- 数据集：`keremberke/hard-hat-detection`（Roboflow Universe Projects），许可证 CC BY 4.0；详见 `DATA.md`。
- 模型架构：torchvision 的 SSDlite320 MobileNetV3-Large（BSD-3-Clause）；使用 COCO 预训练权重，替换分类头后在上述数据集上微调。
- 是否复制代码：否。数据加载、训练循环、评估、标注、Flask 页面、报告版式均为自己编写；只使用 torchvision 提供的模型和算子。
- 第三方库：PyTorch（BSD-3-Clause）、torchvision（BSD-3-Clause）、Flask（BSD-3-Clause）、Pillow（HPND）、NumPy（BSD-3-Clause）。
- 独立实现说明：目录结构、命令行参数、Web 交互、报告样式、中文文案均自行设计。
