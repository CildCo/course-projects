"""重复文件清理器：按内容哈希找出重复文件，可选检测相似图片。

默认只扫描和生成报告，不删除、不移动任何文件。
加 --apply 会把重复文件移到 .dedupe-trash，并生成撤销清单；--undo 可以还原。
"""

import argparse
import hashlib
import html
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
SKIP_DIRS = {
    ".git",
    ".dedupe-trash",
    ".venv",
    "__pycache__",
    "node_modules",
    "output",
    "dist",
    "build",
}
CHUNK_SIZE = 1024 * 1024
MAX_IMAGES = 400


def human_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(max(0, size))
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def iter_files(root: Path, min_size: int):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if set(relative.parts) & SKIP_DIRS:
            continue
        try:
            if path.stat().st_size < min_size:
                continue
        except OSError:
            continue
        yield path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def find_exact_duplicates(root: Path, min_size: int = 1):
    by_size = defaultdict(list)
    for path in iter_files(root, min_size):
        try:
            by_size[path.stat().st_size].append(path)
        except OSError:
            continue

    groups = []
    for size, paths in by_size.items():
        if len(paths) < 2:
            continue
        by_hash = defaultdict(list)
        for path in paths:
            try:
                by_hash[file_hash(path)].append(path)
            except OSError:
                continue
        for digest, same in by_hash.items():
            if len(same) > 1:
                groups.append(
                    {
                        "hash": digest,
                        "size": size,
                        "paths": sorted(str(path.relative_to(root)) for path in same),
                    }
                )
    groups.sort(key=lambda item: item["size"] * (len(item["paths"]) - 1), reverse=True)
    return groups


def dhash(path: Path, hash_size: int = 8) -> int:
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    with Image.open(path) as image:
        gray = image.convert("L").resize((hash_size + 1, hash_size), resampling)
        pixels = list(gray.tobytes())
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def find_similar_images(root: Path, max_distance: int = 6):
    try:
        import PIL  # noqa: F401
    except ImportError:
        return [], "未安装 Pillow，跳过相似图片检测"

    candidates = []
    for path in iter_files(root, 1):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            candidates.append((path, dhash(path)))
        except Exception:
            continue
    if len(candidates) > MAX_IMAGES:
        candidates = candidates[:MAX_IMAGES]
        note = f"图片超过 {MAX_IMAGES} 张，只检测前 {MAX_IMAGES} 张"
    else:
        note = ""

    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for i in range(len(candidates)):
        for j in range(i):
            if (candidates[i][1] ^ candidates[j][1]).bit_count() <= max_distance:
                union(i, j)

    grouped = defaultdict(list)
    for index, (path, _) in enumerate(candidates):
        grouped[find(index)].append(path)

    groups = []
    for paths in grouped.values():
        if len(paths) < 2:
            continue
        groups.append(
            {
                "paths": sorted(str(path.relative_to(root)) for path in paths),
                "size": sum(path.stat().st_size for path in paths),
            }
        )
    groups.sort(key=lambda item: len(item["paths"]), reverse=True)
    return groups, note


def quarantine(root: Path, groups, trash_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = trash_root / stamp
    moved = []
    for group in groups:
        paths = [root / item for item in group["paths"]]
        keep = paths[0]
        for path in paths[1:]:
            relative = path.relative_to(root)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(destination))
            moved.append({"from": str(path), "to": str(destination), "kept": str(keep)})

    manifest = target / "undo.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"root": str(root), "trash": str(target), "moved": moved}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def undo(manifest_file: Path) -> int:
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    restored = 0
    for item in data.get("moved", []):
        source = Path(item["to"])
        target = Path(item["from"])
        if not source.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target = target.with_name(target.name + ".restored")
        shutil.move(str(source), str(target))
        restored += 1
    print(f"已恢复 {restored} 个文件")
    return restored


def create_demo(root: Path) -> None:
    if root.name != "demo":
        raise SystemExit("演示目录名必须是 demo，避免误删其他目录")
    if root.exists():
        shutil.rmtree(root)
    (root / "docs").mkdir(parents=True)
    (root / "archive").mkdir(parents=True)
    (root / "images").mkdir(parents=True)
    (root / "keep").mkdir(parents=True)

    report = "2026 年第三季度运营复盘\n\n收入增长 18%，复购率 32%。\n"
    (root / "docs" / "report.txt").write_text(report, encoding="utf-8")
    (root / "docs" / "report-copy.txt").write_text(report, encoding="utf-8")
    (root / "archive" / "report-backup.txt").write_text(report, encoding="utf-8")
    notes = "下周待办\n- 整理渠道数据\n- 更新报价单\n"
    (root / "docs" / "notes.txt").write_text(notes, encoding="utf-8")
    (root / "docs" / "notes-copy.txt").write_text(notes, encoding="utf-8")
    (root / "keep" / "unique.txt").write_text("这份文件没有重复。\n", encoding="utf-8")

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return

    image = Image.new("RGB", (640, 360), "#dbe7ff")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 80, 300, 280), fill="#2f5fe0")
    draw.ellipse((360, 110, 560, 310), fill="#0e1418")
    draw.text((80, 40), "dedupe demo", fill="#0e1418")
    image.save(root / "images" / "cover.png")
    image.save(root / "images" / "cover-copy.png")
    image.resize((512, 288)).save(root / "images" / "cover-small.png")
    bright = image.point(lambda value: min(255, int(value * 1.06)))
    bright.save(root / "images" / "cover-bright.png")


def write_report(root: Path, exact_groups, image_groups, image_note, stats, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exact_savings = sum(group["size"] * (len(group["paths"]) - 1) for group in exact_groups)

    group_rows = []
    for index, group in enumerate(exact_groups[:50], start=1):
        files = "".join(f"<li>{html.escape(item)}</li>" for item in group["paths"])
        group_rows.append(
            f"""
            <tr>
              <td>{index}</td>
              <td><code>{group['hash'][:12]}</code></td>
              <td>{human_size(group['size'])}</td>
              <td>{len(group['paths'])}</td>
              <td>{human_size(group['size'] * (len(group['paths']) - 1))}</td>
              <td><ul class="files">{files}</ul></td>
            </tr>
            """
        )
    if len(exact_groups) > 50:
        group_rows.append(
            f'<tr><td colspan="6" class="muted">仅展示前 50 组，共 {len(exact_groups)} 组。</td></tr>'
        )

    image_cards = []
    for group in image_groups[:12]:
        thumbs = []
        for item in group["paths"]:
            source = (root / item).resolve()
            relative = os.path.relpath(source, report_path.parent).replace("\\", "/")
            thumbs.append(
                f'<figure><img src="{html.escape(relative)}" alt=""><figcaption>{html.escape(item)}</figcaption></figure>'
            )
        image_cards.append(f'<div class="image-group">{"".join(thumbs)}</div>')

    empty_groups = '<p class="empty">没有发现重复文件。</p>'
    if not group_rows:
        group_table = empty_groups
    else:
        group_table = f"""
        <table>
          <thead><tr><th>#</th><th>内容哈希</th><th>单文件</th><th>数量</th><th>可释放</th><th>文件</th></tr></thead>
          <tbody>{''.join(group_rows)}</tbody>
        </table>
        """

    if not image_cards:
        image_section = f'<p class="empty">{html.escape(image_note or "没有发现相似图片。")}</p>'
    else:
        image_section = f'<p class="muted">{html.escape(image_note)}</p>' + "".join(image_cards)

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>重复文件报告</title>
<style>
  :root {{ --ink:#0e1418; --muted:#667085; --line:#e4e7ec; --blue:#2f5fe0; --bg:#f5f7fb; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; line-height:1.6; }}
  main {{ max-width:1080px; margin:0 auto; padding:40px 24px 72px; }}
  header {{ background:#fff; border:1px solid var(--line); border-radius:18px; padding:28px 32px; }}
  h1 {{ margin:0 0 6px; font-size:28px; letter-spacing:-.02em; }}
  h2 {{ margin:36px 0 14px; font-size:20px; }}
  .muted {{ color:var(--muted); }}
  .cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:22px; }}
  .card {{ background:var(--bg); border-radius:12px; padding:16px 18px; }}
  .card strong {{ display:block; font-size:24px; letter-spacing:-.02em; }}
  .card span {{ color:var(--muted); font-size:13px; }}
  section {{ background:#fff; border:1px solid var(--line); border-radius:18px; padding:24px 28px; margin-top:18px; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th, td {{ text-align:left; padding:10px 8px; border-bottom:1px solid var(--line); vertical-align:top; }}
  th {{ color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  code {{ background:#eef2ff; color:#1d3ea8; padding:2px 6px; border-radius:6px; font-size:12px; }}
  ul.files {{ margin:0; padding-left:18px; }}
  .image-group {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:14px 0 24px; }}
  figure {{ margin:0; background:var(--bg); border-radius:12px; padding:10px; }}
  figure img {{ width:100%; height:120px; object-fit:cover; border-radius:8px; display:block; }}
  figcaption {{ font-size:12px; color:var(--muted); margin-top:8px; word-break:break-all; }}
  .empty {{ color:var(--muted); background:var(--bg); border-radius:12px; padding:18px; }}
  .tip {{ background:#eef4ff; border-left:4px solid var(--blue); padding:14px 16px; border-radius:10px; }}
  footer {{ color:var(--muted); font-size:12px; margin-top:24px; text-align:center; }}
</style>
</head>
<body>
<main>
  <header id="overview">
    <h1>重复文件报告</h1>
    <p class="muted">扫描目录：{html.escape(str(root.resolve()))} · 生成时间：{generated}</p>
    <div class="cards">
      <div class="card"><strong>{stats['files']}</strong><span>扫描文件</span></div>
      <div class="card"><strong>{len(exact_groups)}</strong><span>重复组</span></div>
      <div class="card"><strong>{stats['duplicates']}</strong><span>可清理文件</span></div>
      <div class="card"><strong>{human_size(exact_savings)}</strong><span>可释放空间</span></div>
    </div>
  </header>

  <section id="groups">
    <h2>完全重复</h2>
    <p class="muted">按文件大小分组后计算 SHA-256；同组文件内容完全一致。</p>
    {group_table}
  </section>

  <section id="images">
    <h2>相似图片</h2>
    <p class="muted">使用 dHash 感知哈希，距离越小越相似；仅作参考，删除前请人工确认。</p>
    {image_section}
  </section>

  <section id="usage">
    <h2>怎么处理</h2>
    <div class="tip">
      <p><strong>默认只读。</strong> 不加 --apply 不会移动任何文件。</p>
      <p>确认报告后运行：<code>python main.py 目标目录 --apply</code>，重复文件会进入 <code>.dedupe-trash</code>。</p>
      <p>需要还原时：<code>python main.py --undo .dedupe-trash/时间戳/undo.json</code>。</p>
    </div>
  </section>

  <footer>由 dedupe-cli 生成 · 清理前请先备份重要文件</footer>
</main>
</body>
</html>
"""
    report_path.write_text(document, encoding="utf-8")


def run_scan(root: Path, min_size: int, max_distance: int, include_images: bool):
    exact_groups = find_exact_duplicates(root, min_size)
    image_groups, image_note = ([], "已关闭相似图片检测")
    if include_images:
        image_groups, image_note = find_similar_images(root, max_distance)
    files = sum(1 for _ in iter_files(root, 0))
    duplicates = sum(len(group["paths"]) - 1 for group in exact_groups)
    stats = {"files": files, "duplicates": duplicates}
    return exact_groups, image_groups, image_note, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="按内容哈希查找重复文件，默认只生成报告")
    parser.add_argument("target", nargs="?", default=".", help="要扫描的目录，默认当前目录")
    parser.add_argument("--min-size", type=int, default=1, help="忽略小于该字节数的文件")
    parser.add_argument("--max-distance", type=int, default=6, help="相似图片的汉明距离上限")
    parser.add_argument("--no-images", action="store_true", help="跳过相似图片检测")
    parser.add_argument("--apply", action="store_true", help="把重复文件移入 .dedupe-trash")
    parser.add_argument("--undo", type=Path, help="撤销清单路径")
    parser.add_argument("--report", type=Path, default=Path("output/dedupe-report.html"))
    parser.add_argument("--json", type=Path, default=Path("output/duplicates.json"))
    parser.add_argument("--demo", action="store_true", help="生成演示目录 demo 并扫描")
    args = parser.parse_args()

    if args.undo:
        undo(args.undo)
        return

    root = Path(args.target)
    if args.demo:
        root = Path("demo")
        create_demo(root)
        print(f"已生成演示目录：{root.resolve()}")

    if not root.exists():
        raise SystemExit(f"目录不存在：{root}")

    exact_groups, image_groups, image_note, stats = run_scan(
        root, args.min_size, args.max_distance, not args.no_images
    )
    write_report(root, exact_groups, image_groups, image_note, stats, args.report)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(
            {
                "root": str(root.resolve()),
                "exact": exact_groups,
                "similar_images": image_groups,
                "note": image_note,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    savings = sum(group["size"] * (len(group["paths"]) - 1) for group in exact_groups)
    print(f"扫描文件：{stats['files']}")
    print(f"重复组：{len(exact_groups)}，可清理：{stats['duplicates']} 个，可释放：{human_size(savings)}")
    print(f"相似图片组：{len(image_groups)}")
    print(f"报告：{args.report.resolve()}")
    print(f"清单：{args.json.resolve()}")

    if args.apply and exact_groups:
        manifest = quarantine(root, exact_groups, root / ".dedupe-trash")
        print(f"已隔离重复文件，撤销清单：{manifest.resolve()}")


if __name__ == "__main__":
    main()
