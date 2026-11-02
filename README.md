# Pothole Detection (Image) - YOLOv8 + Flask

Upload a road image in the browser and get the potholes marked with confidence scores.
Based on the open-source "Pothole Detection using python and yolov8" project, updated for the current Ultralytics API.

## Setup
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Put your trained weights as `y8best.pt` in this folder (or download the dedicated pothole weights):
```bash
curl -L -o y8best.pt https://huggingface.co/peterhdd/pothole-detection-yolov8/resolve/main/best.pt
```

## Web app
```bash
python app.py
```
Open http://127.0.0.1:5050, choose an image, adjust the confidence slider (recommended: 0.15 - 0.25), and click **Detect potholes**.

## Command line
```bash
python predict_image.py road.jpg
python predict_image.py road.jpg --conf 0.15 --imgsz 640
python predict_image.py images_folder/ --conf 0.15 --out results
```

## Model Information
The project uses a dedicated YOLOv8 Pothole Detection model trained to detect road potholes accurately with high confidence.

## Files
- `app.py` - Flask backend (`POST /api/detect`)
- `index.html` - frontend web UI
- `predict_image.py` - CLI for single images or folders
