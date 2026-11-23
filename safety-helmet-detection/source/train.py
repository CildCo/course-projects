"""Train a small SSDlite detector for hardhat / no-hardhat.

Dataset format: COCO JSON + images, produced by download_data.py or the
project's bundled subset. The model is torchvision SSDlite320 (BSD-3-Clause),
initialized from COCO weights, with a new 3-class head (background + 2 classes).
"""

import argparse
import json
import random
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parent
CLASSES = ["hardhat", "no-hardhat"]
LABEL_MAP = {0: 1, 1: 2}
IN_CHANNELS = [672, 480, 512, 256, 256, 128]
ANCHORS = [6, 6, 6, 6, 6, 6]


class CocoSubset(Dataset):
    def __init__(self, data_dir: Path, annotation_file: Path, train: bool):
        self.data_dir = data_dir
        self.images_dir = data_dir / "images"
        self.train = train
        payload = json.loads(annotation_file.read_text(encoding="utf-8"))
        self.images = payload["images"]
        self.annotations = defaultdict(list)
        for ann in payload["annotations"]:
            self.annotations[ann["image_id"]].append(ann)
        self.jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)

    def __len__(self):
        return len(self.images)

    def has_label(self, index: int, label: int) -> bool:
        image_id = self.images[index]["id"]
        return any(ann["category_id"] == label for ann in self.annotations[image_id])

    def __getitem__(self, index: int):
        info = self.images[index]
        path = self.images_dir / info["file_name"]
        image = Image.open(path).convert("RGB")
        boxes = []
        labels = []
        for ann in self.annotations[info["id"]]:
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(LABEL_MAP[ann["category_id"]])

        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64)

        if self.train:
            if random.random() < 0.5:
                width = image.width
                image = ImageOps.mirror(image)
                if len(boxes_t):
                    x1 = boxes_t[:, 0].clone()
                    x2 = boxes_t[:, 2].clone()
                    boxes_t[:, 0] = width - x2
                    boxes_t[:, 2] = width - x1
            image = self.jitter(image)

        tensor = transforms.functional.to_tensor(image)
        area = (
            (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
            if len(boxes_t)
            else torch.zeros(0)
        )
        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([info["id"]]),
            "area": area,
            "iscrowd": torch.zeros(len(boxes_t), dtype=torch.int64),
        }
        return tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model():
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    model = ssdlite320_mobilenet_v3_large(weights=weights)
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    model.head.classification_head = SSDLiteClassificationHead(
        IN_CHANNELS, ANCHORS, 3, norm_layer
    )
    return model


def evaluate(model, loader, device, threshold=0.3, nohat_threshold=None, iou_threshold=0.5):
    if nohat_threshold is None:
        nohat_threshold = max(0.2, threshold - 0.1)
    model.eval()
    tp = {1: 0, 2: 0}
    fp = {1: 0, 2: 0}
    fn = {1: 0, 2: 0}
    with torch.no_grad():
        for images, targets in loader:
            outputs = model([image.to(device) for image in images])
            for output, target in zip(outputs, targets):
                gt_boxes = target["boxes"].to(device)
                gt_labels = target["labels"].to(device)
                for label in (1, 2):
                    cut = threshold if label == 1 else nohat_threshold
                    mask = (output["labels"] == label) & (output["scores"] >= cut)
                    pred_boxes = output["boxes"][mask]
                    pred_scores = output["scores"][mask]
                    gt = gt_boxes[gt_labels == label]
                    matched = set()
                    order = torch.argsort(pred_scores, descending=True)
                    for index in order.tolist():
                        if len(gt) == 0:
                            fp[label] += 1
                            continue
                        ious = box_iou(pred_boxes[index].unsqueeze(0), gt)[0]
                        best = int(torch.argmax(ious).item())
                        if ious[best] >= iou_threshold and best not in matched:
                            tp[label] += 1
                            matched.add(best)
                        else:
                            fp[label] += 1
                    fn[label] += len(gt) - len(matched)
    metrics = {}
    for label, name in ((1, "hardhat"), (2, "no-hardhat")):
        precision = tp[label] / max(1, tp[label] + fp[label])
        recall = tp[label] / max(1, tp[label] + fn[label])
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        metrics[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp[label],
            "fp": fp[label],
            "fn": fn[label],
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hardhat detector")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--nohat-repeat", type=int, default=1, help="额外重复 no-hardhat 图片的次数")
    parser.add_argument("--resume", type=Path, help="从已有 checkpoint 继续训练")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "helmet_ssdlite.pt")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    train_base = CocoSubset(args.data_dir, args.data_dir / "train.json", train=True)
    val_base = CocoSubset(args.data_dir, args.data_dir / "val.json", train=False)
    if args.max_train:
        train_base.images = train_base.images[: args.max_train]
    if args.max_val:
        val_base.images = val_base.images[: args.max_val]

    indices = list(range(len(train_base)))
    nohat = [index for index in indices if train_base.has_label(index, 1)]
    train_indices = indices + nohat * max(1, args.nohat_repeat)
    train_set = Subset(train_base, train_indices)
    val_set = Subset(val_base, list(range(len(val_base))))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )

    model = build_model().to(device)
    if args.resume:
        if not args.resume.exists():
            raise SystemExit(f"resume checkpoint 不存在：{args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"resumed from {args.resume}")
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []
    print(
        f"train images {len(train_set)} (base {len(train_base)}, no-hat oversampled {len(nohat)}), "
        f"val images {len(val_set)}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        start = time.time()
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = [image.to(device) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.item())
            if step % 20 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.4f}")

        model.train()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [image.to(device) for image in images]
                targets = [
                    {key: value.to(device) for key, value in target.items()}
                    for target in targets
                ]
                loss_dict = model(images, targets)
                val_loss += float(sum(loss for loss in loss_dict.values()).item())
                val_batches += 1
        val_loss /= max(1, val_batches)
        train_loss = running / max(1, len(train_loader))
        elapsed = time.time() - start
        scheduler.step()
        history.append(
            {"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4), "seconds": round(elapsed, 1)}
        )
        print(
            f"epoch {epoch}: train {train_loss:.4f} val {val_loss:.4f} "
            f"lr {scheduler.get_last_lr()[0]:.6f} {elapsed:.1f}s"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "input_size": 320,
                    "history": history,
                    "dataset": "keremberke/hard-hat-detection (CC BY 4.0)",
                },
                args.output,
            )
            print(f"saved best checkpoint: {args.output}")

    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    metrics = evaluate(model, val_loader, device)
    checkpoint["metrics"] = metrics
    checkpoint["history"] = history
    torch.save(checkpoint, args.output)
    card = {
        "classes": CLASSES,
        "input_size": 320,
        "architecture": "torchvision SSDlite320 MobileNetV3-Large",
        "dataset": "keremberke/hard-hat-detection",
        "dataset_license": "CC BY 4.0",
        "dataset_url": "https://huggingface.co/datasets/keremberke/hard-hat-detection",
        "metrics": metrics,
        "history": history,
    }
    (args.output.parent / "model_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("metrics", metrics)
    print("model", args.output, "size MB", round(args.output.stat().st_size / 1024 / 1024, 2))


if __name__ == "__main__":
    main()
