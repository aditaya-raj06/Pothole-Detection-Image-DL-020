"""Pothole detection web app (image upload) - Flask + Ultralytics YOLO."""
import base64
import os
import threading
import time
from pathlib import Path

# Silence Ultralytics config write permission errors on constrained environments
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.environ.get("MODEL_PATH", str(BASE_DIR / "y8best.pt"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB upload limit

_model = None
_lock = threading.Lock()


def get_model():
    """Load the YOLO model once, on first use."""
    global _model
    with _lock:
        if _model is None:
            if not Path(MODEL_PATH).exists():
                raise FileNotFoundError(
                    f"Model file not found: {MODEL_PATH}. Put y8best.pt next to app.py "
                    "or set the MODEL_PATH environment variable."
                )
            from ultralytics import YOLO  # imported lazily so the UI can start without it

            _model = YOLO(MODEL_PATH)
        return _model


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify(model_found=Path(MODEL_PATH).exists(), model_path=Path(MODEL_PATH).name)


@app.route("/api/detect", methods=["POST"])
def detect():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="No image uploaded."), 400

    try:
        conf = float(request.form.get("conf", 0.15))
    except ValueError:
        return jsonify(error="Invalid confidence value."), 400
    conf = min(max(conf, 0.01), 0.99)

    try:
        imgsz = int(request.form.get("imgsz", 960))
    except ValueError:
        imgsz = 960
    imgsz = min(max(imgsz, 320), 1920)

    mode = request.form.get("mode", "potholes")
    if mode not in {"potholes", "road_damage", "all"}:
        return jsonify(error="Invalid display mode."), 400

    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="File is not a valid image (use JPG, PNG, WEBP or BMP)."), 400

    try:
        model = get_model()
        start = time.perf_counter()
        result = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
        elapsed_ms = round((time.perf_counter() - start) * 1000)

        is_single_class = len(result.names) == 1
        if mode == "potholes":
            indices = [
                i for i, box in enumerate(result.boxes)
                if is_single_class or str(result.names[int(box.cls[0])]).strip().casefold() in {"pothole", "potholes", "0"}
            ]
            result.boxes = result.boxes[indices]
        elif mode == "road_damage":
            road_classes = {"pothole", "potholes", "construction_road", "sand_on_road", "garbage", "0"}
            indices = [
                i for i, box in enumerate(result.boxes)
                if is_single_class or str(result.names[int(box.cls[0])]).strip().casefold() in road_classes
            ]
            result.boxes = result.boxes[indices]

        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = (round(v) for v in box.xyxy[0].tolist())
            detections.append(
                {
                    "label": result.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 3),
                    "box": [x1, y1, x2, y2],
                }
            )
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        annotated = result.plot()  # BGR numpy array with boxes drawn
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return jsonify(error="Could not encode the result image."), 500

        return jsonify(
            count=len(detections),
            detections=detections,
            inference_ms=elapsed_ms,
            image="data:image/jpeg;base64," + base64.b64encode(buf).decode(),
        )
    except Exception as exc:
        app.logger.exception("Detection request failed")
        return jsonify(error=f"Detection failed: {exc}"), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image is too large (max 15 MB)."), 413


def find_available_port(preferred=5050):
    import socket
    for p in [preferred, 5050, 8080, 8000]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return preferred


if __name__ == "__main__":
    requested_port = int(os.environ.get("PORT", 5050))
    port = find_available_port(requested_port)
    print(f"\n🚀 Server running at: http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False)
