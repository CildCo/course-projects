"""跑通演示并检查核心逻辑：EXIF 读取、归档计划、移动、撤销、相册生成。"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import main as app  # noqa: E402


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise SystemExit(f"FAIL {name} {detail}")
    print(f"PASS {name} {detail}".rstrip())


def make_photo(path: Path, stamp: str) -> None:
    from PIL import Image

    image = Image.new("RGB", (160, 100), "#dbe7ff")
    exif = Image.Exif()
    exif[36867] = stamp
    exif[306] = stamp
    image.save(path, exif=exif)


def test_plan() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        make_photo(root / "b.jpg", "2024:05:01 10:00:00")
        make_photo(root / "a.jpg", "2024:05:01 10:00:00")
        make_photo(root / "c.jpg", "2023:08:12 09:30:00")
        photos = app.build_plan(root)
        check("读取三张照片", len(photos) == 3, str(len(photos)))
        check("全部来自 EXIF", all(item["dateSource"] == "exif" for item in photos))
        targets = {
            item["source"]: item["target"].replace("\\", "/") for item in photos
        }
        check("同月进入同一目录", targets["a.jpg"].startswith("2024/05/") and targets["b.jpg"].startswith("2024/05/"))
        check("跨年进入不同目录", targets["c.jpg"].startswith("2023/08/"))
        check("同秒文件不重名", len({targets["a.jpg"], targets["b.jpg"]}) == 2)


def test_apply_and_undo() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        make_photo(root / "one.jpg", "2022:11:03 08:00:00")
        make_photo(root / "two.jpg", "2022:11:03 18:00:00")
        photos = app.build_plan(root)
        manifest = app.apply_plan(root, photos, root / ".photo-trash")
        moved = sorted(str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*.jpg"))
        check("归档后位于年月目录", all(item.startswith("2022/11/") for item in moved), str(moved))
        check("生成撤销清单", manifest.exists())

        app.undo(manifest)
        restored = sorted(path.name for path in root.glob("*.jpg"))
        check("撤销后恢复原文件名", restored == ["one.jpg", "two.jpg"], str(restored))


def test_demo() -> None:
    completed = subprocess.run(
        [sys.executable, "main.py", "--demo"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    check("演示命令退出码 0", completed.returncode == 0, completed.stdout[-300:])
    album = ROOT / "output" / "album.html"
    plan = ROOT / "output" / "plan.json"
    check("生成相册", album.exists() and album.stat().st_size > 1000)
    data = json.loads(plan.read_text(encoding="utf-8"))
    photos = data["photos"]
    check("演示照片数量", len(photos) >= 8, f"{len(photos)} 张")
    check("演示覆盖多个月份", len({item["takenAt"][:7] for item in photos}) >= 4)
    check("有 EXIF 日期", sum(1 for item in photos if item["dateSource"] == "exif") >= 7)
    check("有文件时间兜底", any(item["dateSource"] == "mtime" for item in photos))
    check("相册包含标题", "照片相册" in album.read_text(encoding="utf-8"))


def main() -> None:
    test_plan()
    test_apply_and_undo()
    test_demo()
    print("\n全部检查通过。")


if __name__ == "__main__":
    main()
