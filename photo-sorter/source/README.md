# 照片按日期归档

读取照片 EXIF 拍摄时间，按 `YYYY/MM/` 归档并重命名，生成可浏览的 HTML 相册。默认只预览，不会移动照片；加 `--apply` 后才归档，并且可以撤销。

## 功能

- 读取 EXIF 拍摄时间：优先 `DateTimeOriginal`，其次 `DateTimeDigitized`、`DateTime`。
- 文件时间兜底：没有 EXIF 的照片使用文件修改时间，并在计划里标明来源。
- 按年月归档：移动到 `YYYY/MM/YYYYMMDD-HHMMSS-序号.jpg`。
- HTML 相册：按月份分组，直接打开就能浏览。
- 默认只读：不加 `--apply` 只生成计划和相册预览。
- 可撤销：归档时生成 `.photo-trash/时间戳/undo.json`，一条命令还原。

## 安装

```bash
python -m pip install -r requirements.txt
```

需要 Pillow 读取 EXIF 和生成演示照片。

## 使用

预览归档计划：

```bash
python main.py D:/Photos
```

按计划归档：

```bash
python main.py D:/Photos --apply
```

撤销归档：

```bash
python main.py --undo D:/Photos/.photo-trash/20261021-150000/undo.json
```

运行自带演示：

```bash
python main.py --demo
```

演示会在 `demo/` 生成 8 张带日期的照片，在 `output/` 生成相册和计划。

## 输出

| 文件 | 内容 |
|---|---|
| `output/album.html` | 按月份分组的相册 |
| `output/plan.json` | 每张照片的原路径、目标路径、拍摄时间和日期来源 |
| `.photo-trash/<时间戳>/undo.json` | 归档后的撤销清单 |

## 规则

| 情况 | 处理 |
|---|---|
| 有 `DateTimeOriginal` | 使用 EXIF 时间 |
| 只有 `DateTimeDigitized` / `DateTime` | 依次使用 |
| 没有 EXIF | 使用文件修改时间，计划里标记 `mtime` |
| 同一秒多张照片 | 文件名追加三位序号 |
| 目标文件已存在 | 追加 `-1`，不覆盖 |
| 非图片文件 | 忽略 |

## 常见问题

**会把照片移到别的盘吗？**

不会。目标目录始终在当前扫描目录下；跨盘移动由 `shutil.move` 处理，但速度会慢一些。

**HEIC 照片支持吗？**

默认不支持。Pillow 需要额外插件才能读 HEIC，本工具只处理 JPG、PNG、WebP、BMP、GIF。

**为什么有些照片用文件时间？**

照片被微信、社交软件转发后，EXIF 可能被清除。程序会退回文件修改时间，并在 `plan.json` 里写 `mtime`，方便你核对。

**会删除照片吗？**

不会。`--apply` 只移动文件；撤销清单可以还原。归档前仍建议先备份重要照片。

**视频会一起整理吗？**

不会。当前只处理图片；视频归档可以作为后续练习自己扩展。

## License

MIT，见 `LICENSE.md`。演示照片由脚本生成，不包含真实用户照片。
