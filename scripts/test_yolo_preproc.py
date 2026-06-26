"""Diagnose marine-detector ghost boxes on a high-res clip.

Read-only on the pipeline: this script loads weights/detector_deepfish.pt and runs it
directly on a video clip. It does NOT import or modify realtime.py or the classifier.

It answers two questions, both VISUAL (the clip is for inspecting the symptom only — the
confidence threshold must be decided on the held-out TEST set, not here):

  1a. Aspect ratio: compare YOLO's correct internal letterbox (aspect preserved) against a
      manual square resize (aspect squashed). If squashing inflates box counts, a manual
      resize before YOLO would be a bug. Our pipeline does NOT do that resize (it passes the
      raw frame), so this is confirmatory.

  1b/2. Confidence sweep on the clip: for each conf, count boxes per frame over the whole
      clip and save evenly-spaced annotated key frames to <outdir>/conf_<c>/ so the ghosts
      and the real fish can be eyeballed at each threshold.

Usage:
  python scripts/test_yolo_preproc.py --clip fish_clip_2.mp4 --outdir outputs/preproc_test
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

KEYFRAME_COUNT = 6  # evenly-spaced frames saved per conf for visual comparison


def draw_boxes(frame, boxes):
    out = frame.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        conf = float(b.conf[0])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(out, f"{conf:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    return out


def step_1a_aspect(model, clip, n_frames=8):
    """Letterbox (correct) vs manual square resize (squashed) box counts on sample frames."""
    cap = cv2.VideoCapture(clip)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, n_frames, dtype=int)
    lb_total = sq_total = 0
    print("=== STEP 1a: letterbox vs squashed-square (per sampled frame: lb / squashed) ===")
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        # Correct: hand the raw frame to YOLO, which letterboxes to 640 preserving aspect.
        lb = model.predict(frame, conf=0.25, imgsz=640, verbose=False)[0].boxes
        # Wrong: squash to 640x640 first (destroys aspect ratio), then predict.
        squashed = cv2.resize(frame, (640, 640))
        sq = model.predict(squashed, conf=0.25, imgsz=640, verbose=False)[0].boxes
        lb_total += len(lb)
        sq_total += len(sq)
        print(f"  frame {int(idx):4d}: letterbox={len(lb):2d}  squashed={len(sq):2d}")
    cap.release()
    print(f"  TOTAL over {n_frames} frames: letterbox={lb_total}  squashed={sq_total}")
    print("  (pipeline uses the letterbox path; squashed shown only to test the bug hypothesis)\n")


def step_conf_sweep(model, clip, outdir, conf_list):
    """Whole-clip box counts per conf + saved key frames. Returns {conf: stats}."""
    cap = cv2.VideoCapture(clip)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keyframes = set(np.linspace(0, total - 1, KEYFRAME_COUNT, dtype=int).tolist())
    stats = {}
    print(f"=== STEP 2 (clip half): conf sweep over {total} frames ===")
    for conf in conf_list:
        cdir = Path(outdir) / f"conf_{conf}"
        cdir.mkdir(parents=True, exist_ok=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        per_frame = []
        for idx in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            boxes = model.predict(frame, conf=conf, imgsz=640, verbose=False)[0].boxes
            per_frame.append(len(boxes))
            if idx in keyframes:
                cv2.imwrite(str(cdir / f"frame_{idx:04d}.jpg"), draw_boxes(frame, boxes))
        arr = np.array(per_frame)
        stats[conf] = {
            "total_boxes": int(arr.sum()),
            "mean_per_frame": float(arr.mean()),
            "max_per_frame": int(arr.max()),
            "frames_with_boxes": int((arr > 0).sum()),
        }
        s = stats[conf]
        print(f"  conf={conf}: total_boxes={s['total_boxes']:5d}  "
              f"mean/frame={s['mean_per_frame']:.2f}  max/frame={s['max_per_frame']:3d}  "
              f"frames_with_boxes={s['frames_with_boxes']}/{total}  -> {cdir}")
    cap.release()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="weights/detector_deepfish.pt")
    ap.add_argument("--clip", default="fish_clip_2.mp4")
    ap.add_argument("--outdir", default="outputs/preproc_test")
    ap.add_argument("--conf-list", default="0.25,0.5,0.65,0.8")
    args = ap.parse_args()

    if not Path(args.detector).exists():
        raise SystemExit(f"detector not found: {args.detector}")
    if not Path(args.clip).exists():
        raise SystemExit(f"clip not found: {args.clip}")

    conf_list = [float(c) for c in args.conf_list.split(",")]
    model = YOLO(args.detector)

    step_1a_aspect(model, args.clip)
    stats = step_conf_sweep(model, args.clip, args.outdir, conf_list)

    # Markdown summary of the CLIP side only (test-set metrics live elsewhere — see report).
    lines = [
        "# Clip ghost-box diagnosis — fish_clip_2",
        "",
        "Box counts from `weights/detector_deepfish.pt` on the 4K clip at each conf.",
        "VISUAL ONLY — threshold must be chosen on the held-out TEST set, not here.",
        "",
        "| conf | total boxes | mean/frame | max/frame | frames with boxes |",
        "|---:|---:|---:|---:|---:|",
    ]
    for conf in conf_list:
        s = stats[conf]
        lines.append(f"| {conf} | {s['total_boxes']} | {s['mean_per_frame']:.2f} | "
                     f"{s['max_per_frame']} | {s['frames_with_boxes']} |")
    out_md = Path(args.outdir) / "clip_box_counts.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_md}")


if __name__ == "__main__":
    main()
