"""
Real-time fish detection and species identification.

Pipeline per frame:
  1. YOLOv8n detects all fish → bounding boxes
  2. Each crop → EfficientNet-B0 → species + confidence
  3. OpenCV overlays results and streams to display

Usage:
    python src/realtime.py --source 0               # webcam
    python src/realtime.py --source video.mp4       # video file
    python src/realtime.py --source video.mp4 --save outputs/result.mp4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import build_model
from src.species_map import DISPLAY_NAMES

CLASSIFY_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Minimum detection confidence to attempt species classification
DETECT_CONF = 0.35
# Minimum classifier confidence to display species label
CLASSIFY_CONF = 0.40

# Overlay colours
BOX_COLOR = (0, 200, 100)
TEXT_BG   = (0, 0, 0)
TEXT_FG   = (255, 255, 255)
FPS_COLOR = (100, 220, 100)


def load_classifier(checkpoint: str, classes_path: str, device):
    with open(classes_path) as f:
        class_names = json.load(f)
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    getattr(model, "eval")()
    model.to(device)
    return model, class_names


def classify_crop(model, crop_bgr: np.ndarray, class_names: list, device) -> tuple[str, float]:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(crop_rgb)
    tensor = CLASSIFY_TRANSFORMS(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        conf, idx = probs.max(1)
    folder_name = class_names[idx.item()]
    display = DISPLAY_NAMES.get(folder_name, folder_name.replace("_", " "))
    return display, conf.item()


def draw_detection(frame, x1, y1, x2, y2, label: str, conf: float):
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
    text = f"{label}  {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), TEXT_BG, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_FG, 1, cv2.LINE_AA)


def run(source, checkpoint: str, classes_path: str, save_path: str = None, show: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    detector = YOLO("yolov8n.pt")            # auto-downloads on first run
    classifier, class_names = load_classifier(checkpoint, classes_path, device)

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_src, (w, h))

    fps_counter, fps_display = 0, 0.0
    t_fps = time.time()

    print("Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- detection ---
        results = detector(frame, conf=DETECT_CONF, classes=[0, 14, 15, 16],
                           verbose=False)  # COCO: person excluded; fish-adjacent classes
        # Fall back: detect all classes if no fish-specific model
        if len(results[0].boxes) == 0:
            results = detector(frame, conf=DETECT_CONF, verbose=False)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            crop = frame[y1:y2, x1:x2]
            species, conf = classify_crop(classifier, crop, class_names, device)

            if conf >= CLASSIFY_CONF:
                draw_detection(frame, x1, y1, x2, y2, species, conf)

        # --- FPS overlay ---
        fps_counter += 1
        if time.time() - t_fps >= 1.0:
            fps_display = fps_counter / (time.time() - t_fps)
            fps_counter = 0
            t_fps = time.time()
        cv2.putText(frame, f"FPS {fps_display:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, FPS_COLOR, 2, cv2.LINE_AA)

        if writer:
            writer.write(frame)
        if show:
            cv2.imshow("Fish Species — Real-Time ID", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Real-time fish species identification")
    parser.add_argument("--source", default="0", help="Webcam index or video file path")
    parser.add_argument("--model", default="outputs/best_model.pt")
    parser.add_argument("--classes", default="outputs/classes.json")
    parser.add_argument("--save", default=None, help="Save annotated video to this path")
    parser.add_argument("--no-show", action="store_true", help="Run headless (no display window)")
    args = parser.parse_args()

    run(args.source, args.model, args.classes,
        save_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
