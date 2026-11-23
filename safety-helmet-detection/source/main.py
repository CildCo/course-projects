"""Safety helmet detector: SSDlite320 trained on a CC BY 4.0 hard-hat dataset.

Modes:
    python main.py --demo                 # run sample images, write report + annotated images
    python main.py image.jpg              # detect one image, save annotated result
    python main.py --serve --port 8020    # start the web UI
"""

import argparse
import json
import os
import time
import uuid
from collections import Counter
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torchvision import transforms
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
CHECKPOINT = Path(os.environ.get("HELMET_MODEL", MODELS / "helmet_ssdlite.pt"))
SAMPLES = ROOT / "sample_images"
OUTPUT = ROOT / "output"
CLASSES = ["hardhat", "no-hardhat"]
LABEL_NAME = {1: "hardhat", 2: "no-hardhat"}
LABEL_CN = {1: "已戴安全帽", 2: "未戴安全帽"}
COLORS = {1: (31, 138, 76), 2: (192, 57, 43)}
IN_CHANNELS = [672, 480, 512, 256, 256, 128]
ANCHORS = [6, 6, 6, 6, 6, 6]

_model = None
_model_card = None


def build_model():
    model = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    )
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    model.head.classification_head = SSDLiteClassificationHead(
        IN_CHANNELS, ANCHORS, 3, norm_layer
    )
    return model


def load_model():
    global _model, _model_card
    if _model is not None:
        return _model
    if not CHECKPOINT.exists():
        raise SystemExit(
            f"缺少模型文件：{CHECKPOINT}\n"
            "先把训练好的 helmet_ssdlite.pt 放到 models/，或运行 python train.py。"
        )
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    model = build_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    _model = model
    _model_card = {
        "classes": checkpoint.get("classes", CLASSES),
        "input_size": checkpoint.get("input_size", 320),
        "dataset": checkpoint.get("dataset", "keremberke/hard-hat-detection"),
        "dataset_license": "CC BY 4.0",
        "metrics": checkpoint.get("metrics", {}),
        "history": checkpoint.get("history", []),
    }
    return _model


def model_card():
    load_model()
    return _model_card


def to_tensor(image: Image.Image):
    return transforms.functional.to_tensor(image.convert("RGB"))


def detect(image: Image.Image, threshold: float = 0.5):
    model = load_model()
    tensor = to_tensor(image)
    nohat_threshold = max(0.2, threshold - 0.1)
    with torch.no_grad():
        prediction = model([tensor])[0]
    detections = []
    for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
        value = float(score.item())
        label_id = int(label.item())
        if label_id not in LABEL_NAME:
            continue
        cut = threshold if label_id == 1 else nohat_threshold
        if value < cut:
            continue
        detections.append(
            {
                "label": LABEL_NAME[label_id],
                "label_cn": LABEL_CN[label_id],
                "score": round(value, 4),
                "box": [round(float(item), 1) for item in box.tolist()],
            }
        )
    detections.sort(key=lambda item: item["score"], reverse=True)
    counts = Counter(item["label"] for item in detections)
    return {
        "detections": detections,
        "counts": {
            "hardhat": counts.get("hardhat", 0),
            "no-hardhat": counts.get("no-hardhat", 0),
            "total": len(detections),
        },
        "threshold": threshold,
        "nohat_threshold": nohat_threshold,
    }


def load_font(size: int):
    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf"):
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


def annotate(image: Image.Image, detections):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = load_font(max(14, round(canvas.width / 40)))
    for item in detections:
        x1, y1, x2, y2 = item["box"]
        color = COLORS[1 if item["label"] == "hardhat" else 2]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text = f"{item['label_cn']} {item['score']:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle((x1, max(0, y1 - th - 8), x1 + tw + 10, y1), fill=color)
        draw.text((x1 + 5, max(0, y1 - th - 6)), text, fill="white", font=font)
    return canvas


def report_html(results, summary, card):
    rows = []
    base = OUTPUT / "demo"
    for item in results:
        original = item["original"].relative_to(base).as_posix()
        annotated = item["annotated"].relative_to(base).as_posix()
        counts = item["counts"]
        rows.append(
            f"""
            <section class="image-card">
              <div class="image-head">
                <div><strong>{item['name']}</strong><span>{counts['total']} 个检测框 · 未戴 {counts['no-hardhat']}</span></div>
                <span class="badge {'danger' if counts['no-hardhat'] else 'ok'}">{'发现违规' if counts['no-hardhat'] else '未发现违规'}</span>
              </div>
              <div class="compare">
                <figure><img src="{original}" alt=""><figcaption>原图</figcaption></figure>
                <figure><img src="{annotated}" alt=""><figcaption>检测结果</figcaption></figure>
              </div>
            </section>
            """
        )
    metrics = card.get("metrics", {})
    metric_rows = "".join(
        f"<tr><td>{name}</td><td>{values.get('precision', 0):.3f}</td>"
        f"<td>{values.get('recall', 0):.3f}</td><td>{values.get('f1', 0):.3f}</td></tr>"
        for name, values in metrics.items()
    ) or '<tr><td colspan="4">暂无指标</td></tr>'
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>安全帽检测报告</title>
<style>
  :root {{ --ink:#0e1418; --muted:#667085; --line:#e4e7ec; --blue:#2f5fe0; --bg:#f5f7fb; --red:#c0392b; --green:#1f8a4c; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; line-height:1.6; }}
  main {{ max-width:1120px; margin:0 auto; padding:40px 24px 72px; }}
  header {{ background:#fff; border:1px solid var(--line); border-radius:18px; padding:28px 32px; }}
  h1 {{ margin:0 0 6px; font-size:28px; letter-spacing:-.02em; }}
  h2 {{ margin:34px 0 14px; font-size:20px; }}
  .muted {{ color:var(--muted); }}
  .cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:22px; }}
  .card {{ background:var(--bg); border-radius:12px; padding:16px 18px; }}
  .card strong {{ display:block; font-size:24px; letter-spacing:-.02em; }}
  .card span {{ color:var(--muted); font-size:13px; }}
  section.panel {{ background:#fff; border:1px solid var(--line); border-radius:18px; padding:24px 28px; margin-top:18px; }}
  .image-card {{ border-top:1px solid var(--line); padding:18px 0; }}
  .image-card:first-child {{ border-top:0; }}
  .image-head {{ display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:12px; }}
  .image-head span {{ display:block; color:var(--muted); font-size:12px; }}
  .badge {{ border-radius:999px; padding:4px 10px; font-size:12px; }}
  .badge.ok {{ background:#e6f4ea; color:var(--green); }}
  .badge.danger {{ background:#fde2e1; color:var(--red); }}
  .compare {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
  figure {{ margin:0; background:var(--bg); border-radius:12px; padding:8px; }}
  figure img {{ width:100%; border-radius:8px; display:block; }}
  figcaption {{ color:var(--muted); font-size:12px; padding:8px 2px 0; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th, td {{ text-align:left; padding:10px 8px; border-bottom:1px solid var(--line); }}
  th {{ color:var(--muted); font-weight:600; font-size:12px; }}
  code {{ background:#eef2ff; color:#1d3ea8; padding:2px 6px; border-radius:6px; font-size:12px; }}
  .tip {{ background:#eef4ff; border-left:4px solid var(--blue); padding:14px 16px; border-radius:10px; }}
  footer {{ color:var(--muted); font-size:12px; margin-top:24px; text-align:center; }}
</style>
</head>
<body>
<main>
  <header id="overview">
    <h1>安全帽检测报告</h1>
    <p class="muted">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · 模型：SSDLite320 · 阈值：{summary['threshold']}</p>
    <div class="cards">
      <div class="card"><strong>{summary['images']}</strong><span>检测图片</span></div>
      <div class="card"><strong>{summary['total']}</strong><span>检测框</span></div>
      <div class="card"><strong>{summary['hardhat']}</strong><span>已戴安全帽</span></div>
      <div class="card"><strong>{summary['no-hardhat']}</strong><span>未戴安全帽</span></div>
    </div>
  </header>

  <section class="panel" id="results">
    <h2>检测结果</h2>
    {''.join(rows)}
  </section>

  <section class="panel" id="model">
    <h2>模型与数据</h2>
    <p class="muted">数据集：{card.get('dataset')} · 许可证：{card.get('dataset_license')} · 架构：{card.get('architecture', 'SSDLite320 MobileNetV3')}</p>
    <table>
      <thead><tr><th>类别</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
      <tbody>{metric_rows}</tbody>
    </table>
  </section>

  <section class="panel" id="usage">
    <h2>怎么用</h2>
    <div class="tip">
      <p><strong>命令行：</strong><code>python main.py image.jpg</code></p>
      <p><strong>Web 界面：</strong><code>python main.py --serve</code>，打开 <code>http://127.0.0.1:8020</code></p>
      <p><strong>重新训练：</strong>下载数据集后运行 <code>python train.py --epochs 12</code></p>
      <p>本工具只做辅助检测，不能替代现场安全员；模型对遮挡、逆光、小目标可能漏检。</p>
    </div>
  </section>

  <footer>安全帽检测 · 数据集 CC BY 4.0 · 代码 MIT</footer>
</main>
</body>
</html>
"""
    return document


def save_upload(image: Image.Image, prefix: str):
    folder = OUTPUT / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}-{uuid.uuid4().hex[:8]}.jpg"
    path = folder / name
    image.convert("RGB").save(path, quality=92)
    return path


def run_demo(threshold: float):
    samples = sorted(SAMPLES.glob("*"))
    samples = [path for path in samples if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not samples:
        raise SystemExit(f"示例图片目录为空：{SAMPLES}")
    demo_dir = OUTPUT / "demo"
    (demo_dir / "original").mkdir(parents=True, exist_ok=True)
    (demo_dir / "annotated").mkdir(parents=True, exist_ok=True)
    results = []
    total_counts = Counter()
    for path in samples:
        image = Image.open(path).convert("RGB")
        started = time.time()
        result = detect(image, threshold)
        annotated = annotate(image, result["detections"])
        original_path = demo_dir / "original" / f"{path.stem}.jpg"
        annotated_path = demo_dir / "annotated" / f"{path.stem}.jpg"
        image.save(original_path, quality=92)
        annotated.save(annotated_path, quality=92)
        results.append(
            {
                "name": path.name,
                "original": original_path,
                "annotated": annotated_path,
                "counts": result["counts"],
                "elapsed_ms": round((time.time() - started) * 1000),
            }
        )
        total_counts.update(result["counts"])
    summary = {
        "images": len(results),
        "total": sum(item["counts"]["total"] for item in results),
        "hardhat": sum(item["counts"]["hardhat"] for item in results),
        "no-hardhat": sum(item["counts"]["no-hardhat"] for item in results),
        "threshold": threshold,
        "elapsed_ms": sum(item["elapsed_ms"] for item in results),
    }
    html = report_html(results, summary, model_card())
    (demo_dir / "report.html").write_text(html, encoding="utf-8")
    (demo_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "results": [{k: v for k, v in item.items() if k not in {"original", "annotated"}} for item in results]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"图片 {summary['images']} 张，检测框 {summary['total']} 个，"
        f"未戴安全帽 {summary['no-hardhat']} 个"
    )
    print(f"报告：{demo_dir / 'report.html'}")
    return summary


app = Flask(__name__)


@app.get("/")
def index():
    return INDEX_HTML


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "model": CHECKPOINT.exists(), "classes": CLASSES})


@app.get("/api/model")
def api_model():
    return jsonify(model_card())


@app.get("/samples/<path:name>")
def sample_file(name: str):
    return send_from_directory(SAMPLES, name)


@app.get("/output/<path:name>")
def output_file(name: str):
    return send_from_directory(OUTPUT, name)


@app.post("/api/detect")
def api_detect():
    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "请上传图片"}), 400
    threshold = float(request.form.get("threshold", 0.5))
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"ok": False, "error": "无法读取图片，请换一张 JPG/PNG"}), 400
    started = time.time()
    result = detect(image, threshold)
    annotated = annotate(image, result["detections"])
    original_path = save_upload(image, "original")
    annotated_path = save_upload(annotated, "annotated")
    return jsonify(
        {
            "ok": True,
            "counts": result["counts"],
            "detections": result["detections"],
            "image_url": f"/output/{original_path.relative_to(OUTPUT).as_posix()}",
            "annotated_url": f"/output/{annotated_path.relative_to(OUTPUT).as_posix()}",
            "elapsed_ms": round((time.time() - started) * 1000),
        }
    )


@app.post("/api/detect-sample")
def api_detect_sample():
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", ""))
    path = SAMPLES / name
    if not path.exists():
        return jsonify({"ok": False, "error": "示例图片不存在"}), 404
    threshold = float(payload.get("threshold", 0.5))
    image = Image.open(path).convert("RGB")
    started = time.time()
    result = detect(image, threshold)
    annotated = annotate(image, result["detections"])
    original_path = save_upload(image, "sample-original")
    annotated_path = save_upload(annotated, "sample-annotated")
    return jsonify(
        {
            "ok": True,
            "counts": result["counts"],
            "detections": result["detections"],
            "image_url": f"/output/{original_path.relative_to(OUTPUT).as_posix()}",
            "annotated_url": f"/output/{annotated_path.relative_to(OUTPUT).as_posix()}",
            "elapsed_ms": round((time.time() - started) * 1000),
        }
    )


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>安全帽检测</title>
<style>
  :root { --ink:#0e1418; --muted:#667085; --line:#e4e7ec; --blue:#2f5fe0; --bg:#f5f7fb; --red:#c0392b; --green:#1f8a4c; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; line-height:1.6; }
  main { max-width:1080px; margin:0 auto; padding:40px 24px 72px; }
  header { background:#0e1418; color:#fff; border-radius:22px; padding:32px 36px; }
  header h1 { margin:0 0 8px; font-size:28px; letter-spacing:-.02em; }
  header p { margin:0; color:#b8c0cc; }
  .pill { display:inline-block; margin-top:16px; padding:6px 12px; border-radius:999px; background:#1d2730; color:#8fb2ff; font-size:13px; }
  .panel { background:#fff; border:1px solid var(--line); border-radius:18px; padding:24px 28px; margin-top:18px; }
  h2 { margin:0 0 14px; font-size:18px; }
  .drop { border:2px dashed #c7d2e5; border-radius:16px; padding:34px; text-align:center; background:#fafbff; cursor:pointer; }
  .drop.drag { border-color:var(--blue); background:#eef4ff; }
  .drop strong { display:block; font-size:16px; margin-bottom:4px; }
  .muted { color:var(--muted); }
  button { border:0; border-radius:10px; padding:10px 16px; font:inherit; cursor:pointer; }
  .primary { background:var(--blue); color:#fff; }
  .ghost { background:#eef2ff; color:#1d3ea8; }
  .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-top:14px; }
  input[type=range] { width:180px; }
  .cards { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:16px; }
  .card { background:var(--bg); border-radius:12px; padding:14px 16px; }
  .card strong { display:block; font-size:22px; }
  .card span { color:var(--muted); font-size:12px; }
  .result { display:none; }
  .compare { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  figure { margin:0; background:var(--bg); border-radius:12px; padding:8px; }
  figure img { width:100%; border-radius:8px; display:block; }
  figcaption { color:var(--muted); font-size:12px; padding:8px 2px 0; }
  table { width:100%; border-collapse:collapse; font-size:14px; margin-top:12px; }
  th, td { text-align:left; padding:8px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-size:12px; font-weight:600; }
  .samples { display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }
  .samples button { padding:0; border-radius:10px; overflow:hidden; border:1px solid var(--line); background:#fff; }
  .samples img { width:120px; height:72px; object-fit:cover; display:block; }
  .error { color:var(--red); margin-top:10px; }
  footer { color:var(--muted); font-size:12px; text-align:center; margin-top:24px; }
</style>
</head>
<body>
<main>
  <header>
    <h1>安全帽检测</h1>
    <p>上传工地/厂区照片，检测人员是否佩戴安全帽；模型本地 CPU 推理，图片不上传第三方。</p>
    <div class="pill">SSDLite320 · 数据集 CC BY 4.0 · 默认阈值 0.5</div>
  </header>

  <section class="panel">
    <h2>上传图片</h2>
    <div class="drop" id="drop">
      <strong>拖拽图片到这里，或点击选择</strong>
      <span class="muted">支持 JPG / PNG / WebP，单张图片</span>
      <input id="file" type="file" accept="image/*" hidden>
    </div>
    <div class="row">
      <label class="muted">阈值 <input id="threshold" type="range" min="0.2" max="0.9" step="0.05" value="0.5"></label>
      <span id="thresholdValue" class="muted">0.50</span>
      <button class="primary" id="detectBtn">开始检测</button>
      <span id="status" class="muted"></span>
    </div>
    <div id="error" class="error"></div>
    <div class="samples" id="samples"></div>
  </section>

  <section class="panel result" id="result">
    <h2>检测结果</h2>
    <div class="cards">
      <div class="card"><strong id="cTotal">0</strong><span>检测框</span></div>
      <div class="card"><strong id="cHard">0</strong><span>已戴安全帽</span></div>
      <div class="card"><strong id="cNo">0</strong><span>未戴安全帽</span></div>
      <div class="card"><strong id="cMs">0</strong><span>耗时 ms</span></div>
    </div>
    <div class="compare" style="margin-top:16px">
      <figure><img id="original" alt=""><figcaption>原图</figcaption></figure>
      <figure><img id="annotated" alt=""><figcaption>检测结果</figcaption></figure>
    </div>
    <table>
      <thead><tr><th>类别</th><th>置信度</th><th>位置</th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </section>

  <footer>辅助检测，不替代现场安全员 · 模型对遮挡、逆光、小目标可能漏检</footer>
</main>
<script>
const drop = document.getElementById('drop');
const fileInput = document.getElementById('file');
const threshold = document.getElementById('threshold');
const thresholdValue = document.getElementById('thresholdValue');
const statusEl = document.getElementById('status');
const errorEl = document.getElementById('error');
let currentFile = null;

threshold.addEventListener('input', () => thresholdValue.textContent = Number(threshold.value).toFixed(2));

drop.addEventListener('click', () => fileInput.click());
drop.addEventListener('dragover', (event) => { event.preventDefault(); drop.classList.add('drag'); });
drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
drop.addEventListener('drop', (event) => {
  event.preventDefault(); drop.classList.remove('drag');
  if (event.dataTransfer.files.length) setFile(event.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) setFile(fileInput.files[0]); });

function setFile(file) {
  currentFile = file;
  drop.querySelector('strong').textContent = file.name;
  errorEl.textContent = '';
}

document.getElementById('detectBtn').addEventListener('click', async () => {
  if (!currentFile) { errorEl.textContent = '请先选择一张图片'; return; }
  await runDetect(async () => {
    const form = new FormData();
    form.append('image', currentFile);
    form.append('threshold', threshold.value);
    return fetch('/api/detect', { method: 'POST', body: form });
  });
});

async function runDetect(sender) {
  statusEl.textContent = '检测中…'; errorEl.textContent = '';
  try {
    const response = await sender();
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || '检测失败');
    render(data);
    statusEl.textContent = '完成';
  } catch (error) {
    errorEl.textContent = error.message;
    statusEl.textContent = '';
  }
}

function render(data) {
  document.getElementById('result').style.display = 'block';
  document.getElementById('cTotal').textContent = data.counts.total;
  document.getElementById('cHard').textContent = data.counts.hardhat;
  document.getElementById('cNo').textContent = data.counts['no-hardhat'];
  document.getElementById('cMs').textContent = data.elapsed_ms;
  document.getElementById('original').src = data.image_url;
  document.getElementById('annotated').src = data.annotated_url;
  const rows = data.detections.map(item => `<tr><td>${item.label_cn}</td><td>${item.score.toFixed(3)}</td><td>${item.box.join(', ')}</td></tr>`).join('');
  document.getElementById('rows').innerHTML = rows || '<tr><td colspan="3">没有超过阈值的检测框</td></tr>';
}

async function loadSamples() {
  const response = await fetch('/api/samples');
  const data = await response.json();
  const box = document.getElementById('samples');
  box.innerHTML = '';
  (data.samples || []).forEach(name => {
    const button = document.createElement('button');
    const img = document.createElement('img');
    img.src = '/samples/' + encodeURIComponent(name);
    button.appendChild(img);
    button.title = name;
    button.addEventListener('click', () => {
      currentFile = null;
      drop.querySelector('strong').textContent = '示例：' + name;
      runDetect(async () => fetch('/api/detect-sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, threshold: threshold.value })
      }));
    });
    box.appendChild(button);
  });
}

loadSamples();
fetch('/api/model').then(r => r.json()).then(card => {
  const best = Number(card.best_threshold || 0.5);
  if (best >= 0.2 && best <= 0.9) {
    threshold.value = best;
    thresholdValue.textContent = best.toFixed(2);
  }
}).catch(() => {});
</script>
</body>
</html>
"""


@app.get("/api/samples")
def api_samples():
    names = [
        path.name
        for path in sorted(SAMPLES.glob("*"))
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    return jsonify({"samples": names[:6]})


def main() -> None:
    parser = argparse.ArgumentParser(description="安全帽检测")
    parser.add_argument("image", nargs="?", help="单张图片路径")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT / "annotated.jpg")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()

    if args.serve:
        app.run(host="127.0.0.1", port=args.port, debug=False)
        return
    threshold = args.threshold
    if threshold is None:
        threshold = float(model_card().get("best_threshold", 0.5))
    if args.demo:
        run_demo(threshold)
        return
    if not args.image:
        parser.error("请提供图片路径，或使用 --demo / --serve")
    image = Image.open(args.image).convert("RGB")
    result = detect(image, threshold)
    annotated = annotate(image, result["detections"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(args.output, quality=92)
    print(json.dumps(result["counts"], ensure_ascii=False))
    print(f"结果：{args.output}")


if __name__ == "__main__":
    main()
