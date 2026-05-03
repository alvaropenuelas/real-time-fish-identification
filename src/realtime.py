import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.annotator import Annotator
from src.classifier import FishClassifier

# COCO class 16 = fish — replace with fish-specific weights for better recall
DETECT_CONF = 0.4
MIN_CROP_PX = 32


def _parse_source(raw: str):
    if raw.isdigit():
        return int(raw)
    return raw


def main():
    parser = argparse.ArgumentParser(description="Real-time fish species identification")
    parser.add_argument("--source", default="0", help="Webcam index or video file path")
    parser.add_argument("--weights", default="weights/model.pt", help="Path to classifier weights")
    parser.add_argument("--detector", default="weights/yolo_fish.pt", help="Path to YOLOv8 weights")
    parser.add_argument("--conf", type=float, default=0.5, help="Classifier confidence threshold")
    parser.add_argument("--display-fps", action="store_true", help="Print FPS to terminal every 30 frames")
    args = parser.parse_args()

    source = _parse_source(args.source)

    if isinstance(source, str) and not Path(source).exists():
        print(f"Error: video file not found: {source}", file=sys.stderr)
        sys.exit(1)

    detector = YOLO(args.detector)
    classifier = FishClassifier(args.weights, conf_threshold=args.conf)
    annotator = Annotator()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: cannot open source: {source}", file=sys.stderr)
        sys.exit(1)

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = detector(frame, conf=DETECT_CONF, verbose=False)
            boxes = results[0].boxes

            if len(boxes) == 0:
                out = frame
            else:
                out = frame.copy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_w, x2), min(frame_h, y2)

                    if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    result = classifier.predict(crop)
                    if result:
                        best = result["top3"][0]
                        alts = [(p["label"], p["confidence"]) for p in result["top3"][1:]]
                        out = annotator.draw(
                            out, best["label"], best["confidence"],
                            bbox=(x1, y1, x2, y2),
                            alt_predictions=alts or None,
                        )

            cv2.imshow("Fish Species — Real-Time ID", out)

            frame_count += 1
            if args.display_fps and frame_count % 30 == 0:
                elapsed = time.time() - t0
                print(f"FPS: {frame_count / elapsed:.1f}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
