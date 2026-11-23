# 安全帽佩戴检测

上传工地/厂区照片，检测人员是否佩戴安全帽；输出检测框、已戴/未戴数量、标注图和 HTML 报告。模型在本地 CPU 推理，图片不上传第三方。

## 功能

- 图片上传 / 拖拽 / 示例图片；
- 可调置信度阈值；
- 检测 `hardhat`（已戴）和 `no-hardhat`（未戴）；
- 标注图 + 数量统计 + 检测明细；
- 一键生成 HTML 报告；
- 命令行单图检测；
- Web API：`/api/detect`、`/api/detect-sample`、`/api/model`；
- 可重新训练：下载公开数据集，运行 `train.py`。

## 快速开始

```bash
python -m pip install -r requirements.txt
python main.py --demo
```

`--demo` 会用 `sample_images/` 里的图片生成：

```text
output/demo/report.html
output/demo/summary.json
output/demo/original/*.jpg
output/demo/annotated/*.jpg
```

启动 Web 界面：

```bash
python main.py --serve
```

打开 <http://127.0.0.1:8020>，选择或拖入图片，点“开始检测”。

单张图片：

```bash
python main.py sample_images/xxx.jpg --threshold 0.5
```

## 模型

- 架构：torchvision **SSDLite320 MobileNetV3-Large**；
- 参数量约 2.2M，模型文件约 8.8 MB；
- 初始化：COCO 预训练权重，替换分类头为 3 类（背景 + hardhat + no-hardhat）；
- 训练数据：`keremberke/hard-hat-detection` 的 700 张训练子集 + 150 张验证子集；
- 训练脚本：`train.py`，CPU 可跑；
- 指标：验证集 150 张，hardhat 阈值 0.30、no-hardhat 阈值 0.20；
  - hardhat：P 0.664 / R 0.314 / F1 0.427
  - no-hardhat：P 0.594 / R 0.567 / F1 0.580
- 指标一般，适合做演示、课程和辅助提示，不建议直接用于生产安全监控。完整数值见 `models/model_card.json` 或 `/api/model`。

重新训练：

```bash
python download_data.py
python train.py --epochs 12 --batch-size 8 --lr 0.005
```

## 数据集

- 名称：`keremberke/hard-hat-detection`
- 发布方：Roboflow Universe Projects
- 许可证：**CC BY 4.0**
- 页面：<https://huggingface.co/datasets/keremberke/hard-hat-detection>
- 原始规模：19,745 张图片，标签 `hardhat` / `no-hardhat`
- 本项目子集：700 训练 + 150 验证，图片在 `data/images/`，标注在 `data/train.json` / `data/val.json`
- 完整说明见 `DATA.md`；数据集不打进源码包。

## 目录

```text
source/
├── main.py            推理、标注、报告、Flask Web
├── train.py           SSDlite 训练与评估
├── download_data.py   下载完整数据集并划分
├── run_all.py         演示 + API 自检
├── models/            helmet_ssdlite.pt + model_card.json
├── sample_images/     示例图片（CC BY 4.0）
├── data/              训练数据（运行时，不进包）
└── README.md
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| GET | `/api/model` | 模型类别、数据集、指标 |
| GET | `/api/samples` | 示例图片列表 |
| POST | `/api/detect` | `multipart/form-data`，字段 `image`、`threshold` |
| POST | `/api/detect-sample` | JSON `{name, threshold}` |

返回：

```json
{
  "ok": true,
  "counts": {"hardhat": 3, "no-hardhat": 1, "total": 4},
  "detections": [{"label": "no-hardhat", "label_cn": "未戴安全帽", "score": 0.87, "box": [12, 30, 80, 140]}],
  "image_url": "/output/uploads/...",
  "annotated_url": "/output/uploads/..."
}
```

## 常见问题

**CPU 能跑吗？**

可以。模型只有 2.2M 参数，单张 640×640 图片在普通 CPU 上通常几百毫秒到一两秒。首次启动会加载 PyTorch，稍慢。

**为什么有些安全帽没检测到？**

小目标、遮挡、逆光、帽子和背景颜色接近时会漏检。可以把阈值调低看更多候选，但误报会增加。这个工具只做辅助提示，不能替代现场安全员。

**能检测反光衣吗？**

当前模型只训练了 `hardhat` / `no-hardhat`。反光衣需要另找带反光衣标注的数据集，或在 `train.py` 里扩展类别；不要直接把当前模型当反光衣检测用。

**模型文件多大？**

`models/helmet_ssdlite.pt` 约 8.8 MB，随源码包提供。数据集 49 MB 不放进源码包。

**可以商用吗？**

代码 MIT；数据集 CC BY 4.0，需要保留署名和链接；模型权重由本项目在 CC BY 4.0 数据上训练。商用前请自行做安全评估和合规确认。

## License

代码 MIT，见 `LICENSE.md`。数据集 CC BY 4.0，见 `DATA.md`。
