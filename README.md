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
Put your trained weights as `y8best.pt` in this folder (or set `MODEL_PATH=/path/to/weights.pt`).

## Web app
```
python app.py
```
Open http://localhost:5000, choose an image, select **Potholes only** or **All classes**, set the confidence slider, and click **Detect potholes**.

## Command line
```
python predict_image.py road.jpg
python predict_image.py images_folder/ --conf 0.3 --out results
```

## Files
- `app.py` - Flask backend (`POST /api/detect`)
- `index.html` - frontend
- `predict_image.py` - CLI for single images or folders
