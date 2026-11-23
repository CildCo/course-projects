"""Download the full hard-hat validation split and build train/val COCO files.

The model shipped with this project is already trained. This script is only
needed if you want to retrain on more data.

Dataset: keremberke/hard-hat-detection (Roboflow Universe Projects), CC BY 4.0.
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from zipfile import ZipFile

import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
URL = "https://hf-mirror.com/datasets/keremberke/hard-hat-detection/resolve/main/data/valid.zip"


def download(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r下载 {done / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB", end="")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    if (DATA / "train.json").exists() and not args.force:
        print("data/train.json 已存在；加 --force 重新生成")
        return

    archive = DATA / "valid.zip"
    if not archive.exists():
        download(archive)

    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zip_file:
        zip_file.extractall(raw)

    annotation_file = raw / "_annotations.coco.json"
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    images_dir = DATA / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for image in payload["images"]:
        source = raw / image["file_name"]
        target = images_dir / image["file_name"]
        if source.exists() and not target.exists():
            shutil.copy2(source, target)

    random.Random(42).shuffle(payload["images"])
    split = int(len(payload["images"]) * (1 - args.val_ratio))
    train_images = payload["images"][:split]
    val_images = payload["images"][split:]
    train_ids = {image["id"] for image in train_images}
    val_ids = {image["id"] for image in val_images}

    def subset(images, ids):
        return {
            "info": payload.get("info", {}),
            "licenses": payload.get("licenses", []),
            "categories": payload["categories"],
            "images": images,
            "annotations": [ann for ann in payload["annotations"] if ann["image_id"] in ids],
        }

    (DATA / "train.json").write_text(
        json.dumps(subset(train_images, train_ids), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (DATA / "val.json").write_text(
        json.dumps(subset(val_images, val_ids), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"完成：训练 {len(train_images)} 张，验证 {len(val_images)} 张")
    print("现在可以运行：python train.py --epochs 12")


if __name__ == "__main__":
    main()
