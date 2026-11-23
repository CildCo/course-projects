"""照片按日期归档：读取 EXIF 拍摄时间，按年月归档并重命名，生成 HTML 相册。

默认只生成计划和相册预览，不移动任何文件。
加 --apply 才会把照片移动到 YYYY/MM/，并生成可撤销清单。
"""

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
EXIF_TAGS = (36867, 36868, 306)
SKIP_DIRS = {
    ".git",
    ".photo-trash",
    ".venv",
    "__pycache__",
    "node_modules",
    "output",
    "dist",
    "build",
    "demo",
}


def human_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(max(0, size))
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if set(relative.parts) & SKIP_DIRS:
            continue
        yield path


def read_taken_at(path: Path):
    from PIL import Image

    try:
        with Image.open(path) as image:
            exif = image.getexif()
            for tag in EXIF_TAGS:
                value = exif.get(tag)
                if not value:
                    continue
                text = str(value).strip()
                for pattern in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d"):
                    try:
                        return datetime.strptime(text, pattern), "exif"
                    except ValueError:
                        continue
    except Exception:
        pass
    return datetime.fromtimestamp(path.stat().st_mtime), "mtime"


def build_plan(root: Path):
    counters = defaultdict(int)
    photos = []
    for path in iter_images(root):
        taken_at, source = read_taken_at(path)
        folder = taken_at.strftime("%Y/%m")
        counters[folder] += 1
        name = f"{taken_at.strftime('%Y%m%d-%H%M%S')}-{counters[folder]:03d}{path.suffix.lower()}"
        target = Path(folder) / name
        photos.append(
            {
                "source": str(path.relative_to(root)),
                "target": str(target),
                "takenAt": taken_at.strftime("%Y-%m-%d %H:%M:%S"),
                "dateSource": source,
                "size": path.stat().st_size,
            }
        )
    photos.sort(key=lambda item: (item["takenAt"], item["source"]))
    return photos


def apply_plan(root: Path, photos, trash_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target_root = trash_root / stamp
    moved = []
    for photo in photos:
        source = root / photo["source"]
        target = root / photo["target"]
        if source == target:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target = target.with_name(f"{target.stem}-1{target.suffix}")
        shutil.move(str(source), str(target))
        photo["target"] = str(target.relative_to(root))
        moved.append({"from": str(source), "to": str(target)})

    manifest = target_root / "undo.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"root": str(root), "moved": moved}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def undo(manifest_file: Path) -> int:
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    restored = 0
    for item in reversed(data.get("moved", [])):
        source = Path(item["to"])
        target = Path(item["from"])
        if not source.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target = target.with_name(target.name + ".restored")
        shutil.move(str(source), str(target))
        restored += 1
    print(f"已恢复 {restored} 张照片")
    return restored


def create_demo(root: Path) -> None:
    if root.name != "demo":
        raise SystemExit("演示目录名必须是 demo，避免误删其他目录")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    from PIL import Image, ImageDraw

    samples = [
        ("20230812-103000", (2023, 8, 12, 10, 30), "#dbe7ff", "#2f5fe0"),
        ("20240501-091500", (2024, 5, 1, 9, 15), "#ffe8d6", "#e07a2f"),
        ("20240501-092000", (2024, 5, 1, 9, 20), "#e6f4ea", "#1f8a4c"),
        ("20240818-184500", (2024, 8, 18, 18, 45), "#f3e8ff", "#7a3fc0"),
        ("20240818-185500", (2024, 8, 18, 18, 55), "#e8f0ff", "#2f5fe0"),
        ("20250101-000500", (2025, 1, 1, 0, 5), "#fff4e5", "#c07a00"),
        ("20250720-143000", (2025, 7, 20, 14, 30), "#e6f7ff", "#0077b6"),
    ]
    for index, (name, moment, background, accent) in enumerate(samples):
        image = Image.new("RGB", (720, 480), background)
        draw = ImageDraw.Draw(image)
        draw.rectangle((40, 40, 300, 240), fill=accent)
        draw.ellipse((380, 120, 640, 380), fill=accent)
        draw.text((44, 270), f"photo {index + 1}", fill="#0e1418")
        exif = Image.Exif()
        stamp = f"{moment[0]:04d}:{moment[1]:02d}:{moment[2]:02d} {moment[3]:02d}:{moment[4]:02d}:00"
        exif[36867] = stamp
        exif[306] = stamp
        path = root / f"IMG_{index + 1:04d}.jpg"
        image.save(path, exif=exif)

    fallback = Image.new("RGB", (640, 420), "#f1f3f5")
    draw = ImageDraw.Draw(fallback)
    draw.text((40, 40), "no exif, use file time", fill="#0e1418")
    fallback_path = root / "scan-no-exif.png"
    fallback.save(fallback_path)
    timestamp = datetime(2022, 3, 15, 8, 0).timestamp()
    os.utime(fallback_path, (timestamp, timestamp))


def write_album(root: Path, photos, output: Path, applied: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dates = [item["takenAt"] for item in photos]
    years = sorted({item["takenAt"][:4] for item in photos})
    months = Counter(item["takenAt"][:7] for item in photos)
    total_size = sum(item["size"] for item in photos)

    timeline = "".join(
        f'<li><span>{month}</span><strong>{count}</strong><em>张</em></li>'
        for month, count in sorted(months.items())
    )

    groups = defaultdict(list)
    for item in photos:
        groups[item["takenAt"][:7]].append(item)

    sections = []
    for month, items in sorted(groups.items()):
        cards = []
        for item in items:
            display = item["target"] if applied else item["source"]
            image_path = (root / display).resolve()
            relative = os.path.relpath(image_path, output.parent).replace("\\", "/")
            cards.append(
                f"""
                <figure>
                  <img src="{relative}" alt="" loading="lazy">
                  <figcaption><strong>{item['takenAt'][:10]}</strong><span>{item['source'].split('/')[-1]}</span></figcaption>
                </figure>
                """
            )
        sections.append(
            f"""
            <section class="month">
              <h3>{month} <span>{len(items)} 张</span></h3>
              <div class="grid">{''.join(cards)}</div>
            </section>
            """
        )

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>照片相册</title>
<style>
  :root {{ --ink:#0e1418; --muted:#667085; --line:#e4e7ec; --blue:#2f5fe0; --bg:#f5f7fb; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; line-height:1.6; }}
  main {{ max-width:1120px; margin:0 auto; padding:40px 24px 72px; }}
  header {{ background:#fff; border:1px solid var(--line); border-radius:18px; padding:28px 32px; }}
  h1 {{ margin:0 0 6px; font-size:28px; letter-spacing:-.02em; }}
  h2 {{ margin:34px 0 12px; font-size:20px; }}
  h3 {{ margin:26px 0 12px; font-size:16px; }}
  h3 span {{ color:var(--muted); font-weight:400; font-size:13px; }}
  .muted {{ color:var(--muted); }}
  .cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:22px; }}
  .card {{ background:var(--bg); border-radius:12px; padding:16px 18px; }}
  .card strong {{ display:block; font-size:24px; letter-spacing:-.02em; }}
  .card span {{ color:var(--muted); font-size:13px; }}
  section.panel {{ background:#fff; border:1px solid var(--line); border-radius:18px; padding:24px 28px; margin-top:18px; }}
  ul.timeline {{ list-style:none; padding:0; margin:0; display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }}
  ul.timeline li {{ background:var(--bg); border-radius:10px; padding:12px 14px; display:flex; align-items:baseline; gap:8px; }}
  ul.timeline em {{ color:var(--muted); font-style:normal; font-size:12px; }}
  .grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }}
  figure {{ margin:0; background:var(--bg); border-radius:12px; padding:8px; }}
  figure img {{ width:100%; height:150px; object-fit:cover; border-radius:8px; display:block; background:#e9edf3; }}
  figcaption {{ display:flex; justify-content:space-between; gap:8px; font-size:12px; color:var(--muted); padding:8px 2px 2px; }}
  figcaption strong {{ color:var(--ink); font-weight:600; }}
  figcaption span {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .tip {{ background:#eef4ff; border-left:4px solid var(--blue); padding:14px 16px; border-radius:10px; }}
  code {{ background:#eef2ff; color:#1d3ea8; padding:2px 6px; border-radius:6px; font-size:12px; }}
  footer {{ color:var(--muted); font-size:12px; margin-top:24px; text-align:center; }}
</style>
</head>
<body>
<main>
  <header id="overview">
    <h1>照片相册</h1>
    <p class="muted">扫描目录：{root.resolve()} · 生成时间：{generated}</p>
    <div class="cards">
      <div class="card"><strong>{len(photos)}</strong><span>照片</span></div>
      <div class="card"><strong>{len(years)}</strong><span>年份</span></div>
      <div class="card"><strong>{len(months)}</strong><span>月份</span></div>
      <div class="card"><strong>{human_size(total_size)}</strong><span>总大小</span></div>
    </div>
    <p class="muted">{('最早 ' + min(dates) + ' · 最晚 ' + max(dates)) if dates else '没有找到照片'}</p>
  </header>

  <section class="panel" id="timeline">
    <h2>时间线</h2>
    <ul class="timeline">{timeline}</ul>
  </section>

  <section class="panel" id="grid">
    <h2>按月份浏览</h2>
    {''.join(sections) if sections else '<p class="muted">没有找到照片。</p>'}
  </section>

  <section class="panel" id="usage">
    <h2>怎么归档</h2>
    <div class="tip">
      <p><strong>默认只预览。</strong> 不加 --apply 不会移动照片。</p>
      <p>确认后运行：<code>python main.py 照片目录 --apply</code>，照片会按 <code>YYYY/MM/</code> 归档。</p>
      <p>需要还原时：<code>python main.py --undo .photo-trash/时间戳/undo.json</code>。</p>
    </div>
  </section>

  <footer>由 photo-sorter 生成 · 归档前请先备份重要照片</footer>
</main>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="按 EXIF 日期归档照片，默认只生成预览")
    parser.add_argument("target", nargs="?", default=".", help="照片目录，默认当前目录")
    parser.add_argument("--apply", action="store_true", help="按计划移动照片")
    parser.add_argument("--undo", type=Path, help="撤销清单路径")
    parser.add_argument("--album", type=Path, default=Path("output/album.html"))
    parser.add_argument("--plan", type=Path, default=Path("output/plan.json"))
    parser.add_argument("--demo", action="store_true", help="生成演示照片并归档预览")
    args = parser.parse_args()

    if args.undo:
        undo(args.undo)
        return

    root = Path(args.target)
    if args.demo:
        root = Path("demo")
        create_demo(root)
        print(f"已生成演示照片：{root.resolve()}")

    if not root.exists():
        raise SystemExit(f"目录不存在：{root}")

    photos = build_plan(root)
    applied = False
    if args.apply and photos:
        manifest = apply_plan(root, photos, root / ".photo-trash")
        applied = True
        print(f"已归档 {len(photos)} 张照片，撤销清单：{manifest.resolve()}")

    write_album(root, photos, args.album, applied)
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(
        json.dumps({"root": str(root.resolve()), "photos": photos}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    exif_count = sum(1 for item in photos if item["dateSource"] == "exif")
    print(f"照片：{len(photos)} 张，EXIF 日期：{exif_count} 张，文件时间兜底：{len(photos) - exif_count} 张")
    print(f"月份：{len({item['takenAt'][:7] for item in photos})} 个")
    print(f"相册：{args.album.resolve()}")
    print(f"计划：{args.plan.resolve()}")


if __name__ == "__main__":
    main()
