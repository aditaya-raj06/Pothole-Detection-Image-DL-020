"""Command-line image detection (replaces the old predict.py / test.py).

Examples:
    python predict_image.py road.jpg
    python predict_image.py images_folder/ --conf 0.15 --imgsz 960 --out results
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")

import cv2
from ultralytics import YOLO

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser(description="Pothole detection on images")
    ap.add_argument("source", help="image file or folder of images")
    ap.add_argument("--model", default="y8best.pt", help="path to trained weights")
    ap.add_argument("--conf", type=float, default=0.15, help="confidence threshold (default: 0.15)")
    ap.add_argument("--imgsz", type=int, default=960, help="image size for model inference (default: 960)")
    ap.add_argument(
        "--mode",
        choices=["potholes", "road_damage", "all"],
        default="potholes",
        help="filter classes: 'potholes', 'road_damage' (potholes+construction), or 'all'",
    )
    ap.add_argument("--out", default="output", help="folder for annotated images")
    ap.add_argument("--show", action="store_true", help="open a preview window")
    args = ap.parse_args()

    src = Path(args.source)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS) if src.is_dir() else [src]
    if not files:
        raise SystemExit("No images found.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)

    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            print(f"skip (unreadable): {path}")
            continue
        result = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

        is_single_class = len(result.names) == 1
        if args.mode == "potholes":
            indices = [
                i for i, b in enumerate(result.boxes)
                if is_single_class or str(result.names[int(b.cls[0])]).strip().casefold() in {"pothole", "potholes", "0"}
            ]
            result.boxes = result.boxes[indices]
        elif args.mode == "road_damage":
            road_classes = {"pothole", "potholes", "construction_road", "sand_on_road", "garbage", "0"}
            indices = [
                i for i, b in enumerate(result.boxes)
                if is_single_class or str(result.names[int(b.cls[0])]).strip().casefold() in road_classes
            ]
            result.boxes = result.boxes[indices]

        annotated = result.plot()
        save_path = out_dir / f"{path.stem}_detected.jpg"
        cv2.imwrite(str(save_path), annotated)
        labels = [result.names[int(b.cls[0])] for b in result.boxes]
        print(f"{path.name}: {len(result.boxes)} detection(s) {labels} -> {save_path}")
        if args.show:
            cv2.imshow("Pothole Detection", annotated)
            cv2.waitKey(0)
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
