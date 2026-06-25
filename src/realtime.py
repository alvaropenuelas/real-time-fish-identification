import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.annotator import Annotator
from src.classifier import FishClassifier
from src.species_map import DISPLAY_NAMES

# COCO class 16 = fish — replace with fish-specific weights for better recall
DETECT_CONF = 0.4
MIN_CROP_PX = 32


def _parse_source(raw: str):
    if raw.isdigit():
        return int(raw)
    return raw


def run_profile(
    detector, classifier, cap, frame_w, frame_h, max_frames=0, metrics_path="outputs/metrics.json"
):
    """Instrument the existing detect->classify pipeline. No GUI, no draw,
    inference logic identical to main(). Reports per-stage timing.

    Warmup: the first frame triggers lazy CUDA/graph init and is timed but
    excluded from the reported means (recorded separately as warmup_ms).
    """
    detect_ms_total = 0.0
    classify_ms_total = 0.0
    n_crops_total = 0
    frame_count = 0
    warmup_detect_ms = None
    warmup_classify_ms = None
    t_wall0 = None  # start wall clock AFTER warmup frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_d0 = time.perf_counter()
        results = detector(frame, conf=DETECT_CONF, verbose=False)
        detect_ms = (time.perf_counter() - t_d0) * 1000.0
        boxes = results[0].boxes

        frame_classify_ms = 0.0
        frame_crops = 0
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)
            if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
                continue
            crop = frame[y1:y2, x1:x2]
            t_c0 = time.perf_counter()
            classifier.predict(crop)
            frame_classify_ms += (time.perf_counter() - t_c0) * 1000.0
            frame_crops += 1

        if frame_count == 0:
            # warmup frame — record but exclude from means
            warmup_detect_ms = detect_ms
            warmup_classify_ms = frame_classify_ms
            t_wall0 = time.perf_counter()
        else:
            detect_ms_total += detect_ms
            classify_ms_total += frame_classify_ms
            n_crops_total += frame_crops

        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break

    wall_s = time.perf_counter() - t_wall0 if t_wall0 is not None else 0.0
    measured = max(frame_count - 1, 1)  # frames excluding warmup
    metrics = {
        "note": (
            "CPU, non-representative — local Mac has no CUDA; FPS not indicative of "
            "GPU target. Accuracy (see classifier_eval) is valid; latency is not."
        ),
        "device": "cpu",
        "frames_total": frame_count,
        "frames_measured": measured,
        "warmup_detect_ms": round(warmup_detect_ms, 2) if warmup_detect_ms is not None else None,
        "warmup_classify_ms": round(warmup_classify_ms, 2) if warmup_classify_ms is not None else None,
        "end_to_end_fps": round(measured / wall_s, 2) if wall_s > 0 else None,
        "detect_ms_per_frame": round(detect_ms_total / measured, 2),
        "classify_ms_per_frame": round(classify_ms_total / measured, 2),
        "mean_crops_per_frame": round(n_crops_total / measured, 3),
        "mean_classify_ms_per_crop": round(classify_ms_total / n_crops_total, 2) if n_crops_total else None,
    }

    out_path = Path(metrics_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing.setdefault("phase0_baseline", {})["fps_cpu_non_representative"] = metrics
    out_path.write_text(json.dumps(existing, indent=2))

    print("=== PROFILE (warmup frame excluded from means) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"written to {out_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Real-time fish species identification")
    parser.add_argument("--source", default="0", help="Webcam index or video file path")
    parser.add_argument("--weights", default="weights/model.pt", help="Path to classifier weights")
    parser.add_argument("--detector", default="weights/yolo_fish.pt", help="Path to YOLOv8 weights")
    parser.add_argument("--conf", type=float, default=0.5, help="Classifier confidence threshold")
    parser.add_argument("--display-fps", action="store_true", help="Print FPS to terminal every 30 frames")
    parser.add_argument(
        "--save",
        default=None,
        help="Write annotated frames to this mp4 instead of showing a window (headless)",
    )
    parser.add_argument(
        "--predict-log",
        default=None,
        help="Write one CSV row per classified detection (raw top1/top2, emitted flag) to this path",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile detect/classify stages over the source (no GUI); writes outputs/metrics.json",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Profile mode: cap measured frames (0 = whole video)"
    )
    parser.add_argument(
        "--reclassify-interval",
        type=int,
        default=15,
        help="Re-run the classifier on a tracked crop every N frames (else reuse cache)",
    )
    parser.add_argument(
        "--ema-beta",
        type=float,
        default=0.6,
        help="EMA weight on the previous cached softmax (higher = smoother, slower to switch)",
    )
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

    if args.profile:
        try:
            run_profile(detector, classifier, cap, frame_w, frame_h, max_frames=args.max_frames)
        finally:
            cap.release()
        return

    frame_count = 0
    t0 = time.time()

    # Headless save: write annotated frames to mp4, no display window.
    writer = None
    if args.save:
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        out_fps = src_fps if src_fps and src_fps > 0 else 25.0
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, out_fps, (frame_w, frame_h))
        if not writer.isOpened():
            print(f"Error: cannot open VideoWriter for: {args.save}", file=sys.stderr)
            cap.release()
            sys.exit(1)

    # Optional prediction log: one row per classified detection (raw top1/top2 + emitted flag).
    # Read-only on the pipeline — it reports the same softmax vector that drives the drawn label.
    pred_log_file = None
    pred_log = None
    if args.predict_log:
        Path(args.predict_log).parent.mkdir(parents=True, exist_ok=True)
        pred_log_file = open(args.predict_log, "w", newline="")
        pred_log = csv.writer(pred_log_file)
        pred_log.writerow(
            [
                "frame_idx", "track_id", "det_conf",
                "classifier_top1_name", "classifier_top1_conf",
                "classifier_top2_name", "classifier_top2_conf", "emitted",
            ]
        )

    def _name(idx):
        folder = classifier.class_names[idx]
        return DISPLAY_NAMES.get(folder, folder.replace("_", " "))

    # Per-track state: track_id -> {"ema": softmax vector, "last_frame": frame last classified}.
    # A crop is classified only when its track is new or every --reclassify-interval frames;
    # otherwise we reuse the EMA-smoothed cached vector. This cuts classifier calls and the
    # EMA suppresses label flicker frame-to-frame.
    N = max(1, args.reclassify_interval)
    beta = args.ema_beta
    cache = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = detector.track(
                frame, persist=True, tracker="bytetrack.yaml", conf=DETECT_CONF, verbose=False
            )
            boxes = results[0].boxes
            out = frame if len(boxes) == 0 else frame.copy()

            # Collect valid boxes with their track ids and crops.
            items = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_w, x2), min(frame_h, y2)
                if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
                    continue
                tid = int(box.id[0]) if box.id is not None else None
                det_conf = float(box.conf[0]) if box.conf is not None else None
                items.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "tid": tid,
                        "crop": frame[y1:y2, x1:x2],
                        "probs": None,
                        "det_conf": det_conf,
                    }
                )

            # Lazy: classify only new/stale tracks (and untracked boxes), batched in one pass.
            need = [
                i
                for i, it in enumerate(items)
                if it["tid"] is None
                or it["tid"] not in cache
                or frame_count - cache[it["tid"]]["last_frame"] >= N
            ]
            if need:
                batch_probs = classifier.predict_probs_batch([items[i]["crop"] for i in need])
                for i, probs in zip(need, batch_probs, strict=True):
                    tid = items[i]["tid"]
                    if tid is None:
                        items[i]["probs"] = probs  # transient, not cached
                    elif tid in cache:
                        cache[tid]["ema"] = beta * cache[tid]["ema"] + (1.0 - beta) * probs
                        cache[tid]["last_frame"] = frame_count
                    else:
                        cache[tid] = {"ema": probs.copy(), "last_frame": frame_count}

            # Draw every box from its cached (EMA) vector, or transient probs if untracked.
            for it in items:
                probs = it["probs"] if it["tid"] is None else cache[it["tid"]]["ema"]
                if probs is None:
                    continue

                # Log raw top1/top2 from the SAME vector that drives the label, including
                # detections below the floor (emitted=false). Does not affect drawing.
                if pred_log is not None:
                    order = np.argsort(probs)[::-1]
                    i1, i2 = int(order[0]), int(order[1])
                    c1, c2 = float(probs[i1]), float(probs[i2])
                    pred_log.writerow(
                        [
                            frame_count,
                            it["tid"],
                            round(it["det_conf"], 4) if it["det_conf"] is not None else "",
                            _name(i1), round(c1, 4),
                            _name(i2), round(c2, 4),
                            str(c1 >= classifier.threshold).lower(),
                        ]
                    )

                result = classifier.decode(probs)
                if result:
                    best = result["top3"][0]
                    alts = [(p["label"], p["confidence"]) for p in result["top3"][1:]]
                    out = annotator.draw(
                        out,
                        best["label"],
                        best["confidence"],
                        bbox=it["bbox"],
                        alt_predictions=alts or None,
                    )

            # Drop tracks that have not been reclassified recently (left the frame).
            if frame_count % 300 == 0 and cache:
                stale = [tid for tid, v in cache.items() if frame_count - v["last_frame"] > 300]
                for tid in stale:
                    del cache[tid]

            if writer is not None:
                writer.write(out)
            else:
                cv2.imshow("Fish Species — Real-Time ID", out)

            frame_count += 1
            if args.display_fps and frame_count % 30 == 0:
                elapsed = time.time() - t0
                print(f"FPS: {frame_count / elapsed:.1f}")

            if writer is None and cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            elapsed = time.time() - t0
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            print(f"saved {frame_count} annotated frames to {args.save} | {fps:.2f} FPS ({elapsed:.1f}s)")
        if pred_log_file is not None:
            pred_log_file.close()
            print(f"prediction log written to {args.predict_log}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
