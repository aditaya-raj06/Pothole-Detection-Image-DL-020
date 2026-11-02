"""Pothole detection web app (image upload) - Flask + Ultralytics YOLO."""
import base64
import os
import threading
import time
from pathlib import Path

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
        conf = float(request.form.get("conf", 0.25))
    except ValueError:
        return jsonify(error="Invalid confidence value."), 400
    conf = min(max(conf, 0.01), 0.99)

    mode = request.form.get("mode", "potholes")
    if mode not in {"potholes", "all"}:
        return jsonify(error="Invalid display mode."), 400

    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="File is not a valid image (use JPG, PNG, WEBP or BMP)."), 400

    try:
        model = get_model()
        start = time.perf_counter()
        result = model.predict(img, conf=conf, verbose=False)[0]
        elapsed_ms = round((time.perf_counter() - start) * 1000)

        if mode == "potholes":
            pothole_indices = [
                i for i, box in enumerate(result.boxes)
                if str(result.names[int(box.cls[0])]).strip().casefold() in {"pothole", "potholes"}
            ]
            result.boxes = result.boxes[pothole_indices]

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
