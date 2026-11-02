"""Command-line image detection (replaces the old predict.py / test.py).

Examples:
    python predict_image.py road.jpg
    python predict_image.py images_folder/ --conf 0.3 --out results
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser(description="Pothole detection on images")
    ap.add_argument("source", help="image file or folder of images")
    ap.add_argument("--model", default="y8best.pt", help="path to trained weights")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
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
        result = model.predict(img, conf=args.conf, verbose=False)[0]
        annotated = result.plot()
        save_path = out_dir / f"{path.stem}_detected.jpg"
        cv2.imwrite(str(save_path), annotated)
        print(f"{path.name}: {len(result.boxes)} pothole(s) -> {save_path}")
        if args.show:
            cv2.imshow("Pothole Detection", annotated)
            cv2.waitKey(0)
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
