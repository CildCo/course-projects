# 重复文件清理器

按内容哈希找出重复文件，支持图片相似检测、HTML 报告、隔离和撤销。默认只读，不会直接删除文件。

## 功能

- 完全重复：先按文件大小分组，再计算 SHA-256，避免对每个文件都做完整哈希。
- 相似图片：dHash 感知哈希 + 汉明距离，找出缩放、轻微调色后的图片。
- 默认只读：不加 `--apply` 只生成报告，不移动、不删除任何文件。
- 可撤销：`--apply` 会把重复文件移入 `.dedupe-trash/时间戳/`，同时生成 `undo.json`。
- HTML 报告：概览、重复组、相似图片、使用说明，可直接在浏览器打开。

## 安装

```bash
python -m pip install -r requirements.txt
```

核心的完全重复检测只用标准库；`Pillow` 只用于相似图片检测和演示图片生成。

## 使用

扫描当前目录，生成报告：

```bash
python main.py .
```

扫描指定目录，忽略小于 1 MB 的文件：

```bash
python main.py D:/Downloads --min-size 1048576
```

确认报告后隔离重复文件：

```bash
python main.py D:/Downloads --apply
```

撤销上一次隔离：

```bash
python main.py --undo D:/Downloads/.dedupe-trash/20261021-143000/undo.json
```

运行自带演示：

```bash
python main.py --demo
```

演示会在 `demo/` 生成重复文档和相似图片，并在 `output/` 生成报告。

## 输出

| 文件 | 内容 |
|---|---|
| `output/dedupe-report.html` | 可视化报告 |
| `output/duplicates.json` | 重复组和相似图片清单 |
| `.dedupe-trash/<时间戳>/undo.json` | 隔离后的撤销清单 |

## 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `target` | `.` | 要扫描的目录 |
| `--min-size` | `1` | 忽略小于该字节数的文件 |
| `--max-distance` | `6` | 相似图片的汉明距离上限，越大越宽松 |
| `--no-images` | 关闭 | 跳过相似图片检测 |
| `--apply` | 关闭 | 把重复文件移入 `.dedupe-trash` |
| `--undo` | 无 | 按撤销清单还原文件 |
| `--report` | `output/dedupe-report.html` | 报告路径 |
| `--json` | `output/duplicates.json` | 清单路径 |

## 目录

```text
source/
├── main.py            扫描、哈希、相似图片、报告、隔离和撤销
├── run_all.py         演示 + 核心逻辑自检
├── requirements.txt   Pillow（可选）
└── README.md
```

## 常见问题

**会误删文件吗？**

不会直接删除。默认只读；即使加 `--apply`，文件也只是被移动到 `.dedupe-trash`，可以用 `--undo` 还原。清理前仍建议先备份重要目录。

**相似图片为什么没有全部找出来？**

dHash 只比较结构相似度，对旋转、裁剪、大幅调色的图片不敏感。把 `--max-distance` 调大可以放宽，但误报也会增加，删除前请人工确认。

**硬链接会被当成重复文件吗？**

会被内容哈希识别为同一组，但硬链接本身不占额外空间。移动硬链接前请自行确认；本工具不做特殊处理。

**Windows 中文路径可以吗？**

可以。程序使用 `pathlib`，读写文件都按 UTF-8 处理。

## License

MIT，见 `LICENSE.md`。演示数据由脚本生成，不包含真实用户文件。
