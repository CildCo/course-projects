"""Run the demo and check inference, report generation and the web API."""

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import main as app  # noqa: E402


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise SystemExit(f"FAIL {name} {detail}")
    print(f"PASS {name} {detail}".rstrip())


def main() -> None:
    check("模型文件存在", app.CHECKPOINT.exists(), str(app.CHECKPOINT))
    card = app.model_card()
    check("模型类别", card["classes"] == ["hardhat", "no-hardhat"])
    check("模型指标", bool(card.get("metrics")))

    threshold = float(card.get("best_threshold", 0.5))
    summary = app.run_demo(threshold)
    check("演示有检测框", summary["total"] > 0, f"{summary['total']} 个")
    check("报告生成", (app.OUTPUT / "demo" / "report.html").exists())
    check("摘要生成", (app.OUTPUT / "demo" / "summary.json").exists())
    annotated = list((app.OUTPUT / "demo" / "annotated").glob("*.jpg"))
    check("标注图生成", len(annotated) >= 3, f"{len(annotated)} 张")

    samples = [path.name for path in app.SAMPLES.glob("*") if path.suffix.lower() in {".jpg", ".png", ".jpeg"}]
    check("示例图片", len(samples) > 0, str(len(samples)))

    client = app.app.test_client()
    health = client.get("/api/health").get_json()
    check("健康接口", health["ok"] and health["model"])

    model = client.get("/api/model").get_json()
    check("模型接口", model["classes"] == ["hardhat", "no-hardhat"])

    sample_response = client.post(
        "/api/detect-sample", json={"name": samples[0], "threshold": threshold}
    )
    sample_data = sample_response.get_json()
    check("示例检测接口", sample_data.get("ok") is True, sample_data.get("error", ""))
    check("示例检测有结果", sample_data["counts"]["total"] >= 0)
    check("示例检测标注图", bool(sample_data["annotated_url"]))

    image_path = app.SAMPLES / samples[0]
    with image_path.open("rb") as handle:
        upload = client.post(
            "/api/detect",
            data={"image": (io.BytesIO(handle.read()), samples[0]), "threshold": str(threshold)},
            content_type="multipart/form-data",
        )
    upload_data = upload.get_json()
    check("上传检测接口", upload_data.get("ok") is True, upload_data.get("error", ""))
    check("上传检测返回", "counts" in upload_data and "detections" in upload_data)

    print("\n全部检查通过。")


if __name__ == "__main__":
    main()
