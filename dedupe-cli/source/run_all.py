"""跑通演示并检查核心逻辑：哈希分组、隔离、撤销、报告生成。"""

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


def test_exact_duplicates() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "a.txt").write_text("same content", encoding="utf-8")
        (root / "b.txt").write_text("same content", encoding="utf-8")
        (root / "c.txt").write_text("different", encoding="utf-8")
        groups = app.find_exact_duplicates(root)
        check("找出完全重复", len(groups) == 1)
        check("重复组包含两个文件", len(groups[0]["paths"]) == 2, str(groups[0]["paths"]))

        manifest = app.quarantine(root, groups, root / ".dedupe-trash")
        remaining = sorted(path.name for path in root.glob("*.txt"))
        check("隔离后只保留一个副本", remaining == ["a.txt", "c.txt"], str(remaining))
        check("生成撤销清单", manifest.exists())

        app.undo(manifest)
        restored = sorted(path.name for path in root.glob("*.txt"))
        check("撤销后两个文件都回来", restored == ["a.txt", "b.txt", "c.txt"], str(restored))


def test_demo() -> None:
    completed = subprocess.run(
        [sys.executable, "main.py", "--demo"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    check("演示命令退出码 0", completed.returncode == 0, completed.stdout[-300:])
    report = ROOT / "output" / "dedupe-report.html"
    payload = ROOT / "output" / "duplicates.json"
    check("生成 HTML 报告", report.exists() and report.stat().st_size > 1000)
    data = json.loads(payload.read_text(encoding="utf-8"))
    check("演示数据包含重复组", len(data["exact"]) >= 2, f"{len(data['exact'])} 组")
    check("演示数据包含相似图片", len(data["similar_images"]) >= 1, f"{len(data['similar_images'])} 组")
    check("报告包含概览", "重复文件报告" in report.read_text(encoding="utf-8"))


def main() -> None:
    test_exact_duplicates()
    test_demo()
    print("\n全部检查通过。")


if __name__ == "__main__":
    main()
