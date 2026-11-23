"""Evaluate a trained checkpoint across confidence thresholds and update the model card."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import train as training

ROOT = Path(__file__).resolve().parent
THRESHOLDS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "helmet_ssdlite.pt")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = training.build_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    val_base = training.CocoSubset(args.data_dir, args.data_dir / "val.json", train=False)
    loader = DataLoader(
        val_base,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=training.collate_fn,
        drop_last=False,
    )

    sweep = {}
    print(f"{'hardhat t':>9} | {'no-hat t':>8} | {'hardhat P/R/F1':>24} | {'no-hardhat P/R/F1':>24} | macro F1")
    for threshold in THRESHOLDS:
        nohat_threshold = max(0.2, threshold - 0.1)
        metrics = training.evaluate(
            model,
            loader,
            torch.device("cpu"),
            threshold=threshold,
            nohat_threshold=nohat_threshold,
        )
        hard = metrics["hardhat"]
        nohat = metrics["no-hardhat"]
        macro = (hard["f1"] + nohat["f1"]) / 2
        sweep[str(threshold)] = {
            "metrics": metrics,
            "nohat_threshold": nohat_threshold,
            "macro_f1": round(macro, 4),
        }
        print(
            f"{threshold:9.2f} | {nohat_threshold:8.2f} | "
            f"{hard['precision']:.3f}/{hard['recall']:.3f}/{hard['f1']:.3f} | "
            f"{nohat['precision']:.3f}/{nohat['recall']:.3f}/{nohat['f1']:.3f} | "
            f"{macro:.4f}"
        )

    best_threshold = max(sweep, key=lambda key: sweep[key]["macro_f1"])
    best = sweep[best_threshold]
    checkpoint["metrics"] = best["metrics"]
    checkpoint["best_threshold"] = float(best_threshold)
    checkpoint["nohat_threshold"] = best["nohat_threshold"]
    checkpoint["threshold_sweep"] = sweep
    torch.save(checkpoint, args.checkpoint)

    card_path = args.checkpoint.parent / "model_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
    card.update(
        {
            "metrics": best["metrics"],
            "best_threshold": float(best_threshold),
            "nohat_threshold": best["nohat_threshold"],
            "threshold_sweep": sweep,
        }
    )
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.checkpoint.parent / "evaluation.json").write_text(
        json.dumps({"best_threshold": best_threshold, "sweep": sweep}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"best threshold {best_threshold}, macro F1 {best['macro_f1']}")


if __name__ == "__main__":
    main()
