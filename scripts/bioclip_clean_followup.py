"""BioCLIP follow-up: full score distributions + more clean crops. Closes the investigation.

Reuses the existing crops and the same scheme A/B label lists (with-clarkii only). No new
architecture. Two phases:

  extract : (STEP 1) score the existing 10 crops dumping the FULL ranked distribution, and
            (STEP 2a) cut candidate crops from long-lived tracks at several frames so they
            can be eyeballed for genuine CLEAN_FISH content.
  score   : (STEP 2b) score a hand-confirmed list of clean crops, both schemes, full dist.

torch.compile is patched to eager (pybioclip calls it; torch 2.2 Dynamo is unsupported on
Python 3.12).
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
from PIL import Image

CLIP = "fish_clip_2.mp4"
DETECTOR = "weights/detector_deepfish.pt"
PRED_CSV = "outputs/demo_fish_clip_2_predictions.csv"
OUTDIR = Path("outputs/bioclip_clean")

SCHEME_A = ["Amphiprion clarkii", "Amphiprion ocellaris", "Amphiprion percula",
            "Pomacentrus moluccensis", "Dascyllus reticulatus", "Chromis chrysura",
            "Sea anemone", "Coral reef"]
SCHEME_B = ["Clark's anemonefish", "Common clownfish", "Orange clownfish",
            "Lemon damselfish", "Reticulated dascyllus", "Pacific chromis",
            "Sea anemone", "Coral reef"]
SCHEMES = {"A": SCHEME_A, "B": SCHEME_B}
CLARKII = {"A": "Amphiprion clarkii", "B": "Clark's anemonefish"}

# frames already used by the original easy/hard set — skip so candidates are NEW segments.
USED_FRAMES = {165, 37, 310, 369, 182, 351, 116, 217, 253, 275}


def score_crops(paths):
    import torch
    torch.compile = lambda model=None, *a, **k: model
    from bioclip import CustomLabelsClassifier

    rows = []
    for sk, labels in SCHEMES.items():
        clf = CustomLabelsClassifier(cls_ary=labels, device="cpu")
        preds = clf.predict(paths, k=None)  # flat {file_name, classification, score}
        by_file = defaultdict(list)
        for pr in preds:
            by_file[pr["file_name"]].append((pr["classification"], float(pr["score"])))
        for fn, sc in by_file.items():
            sc.sort(key=lambda x: x[1], reverse=True)
            ranked = [lbl for lbl, _ in sc]
            clk = CLARKII[sk]
            rows.append({
                "crop": Path(fn).stem, "scheme": sk,
                "top1_label": sc[0][0], "top1_score": round(sc[0][1], 4),
                "clarkii_rank": ranked.index(clk) + 1,
                "clarkii_score": round(dict(sc)[clk], 4),
                "clarkii_is_top1": sc[0][0] == clk,
                "full_dist": json.dumps([[lbl, round(s, 4)] for lbl, s in sc]),
            })
    return rows


def cut_crop(model, cap, frame_idx, target_conf, out_path):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    boxes = model.predict(frame, conf=0.1, imgsz=640, verbose=False)[0].boxes
    if len(boxes) == 0:
        return None
    best = min(boxes, key=lambda b: abs(float(b.conf[0]) - target_conf))
    x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
    x1, y1 = max(0, x1), max(0, y1)
    crop = frame[y1:y2, x1:x2]
    Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(out_path)
    return (x2 - x1, y2 - y1, float(best.conf[0]))


def extract_candidates(tracks, per_track=3):
    """Cut candidate crops from the highest-conf NEW frames of each given track."""
    from ultralytics import YOLO
    rows = list(csv.DictReader(open(PRED_CSV)))
    by_track = defaultdict(list)
    for r in rows:
        if r["track_id"]:
            by_track[r["track_id"]].append((int(r["frame_idx"]), float(r["det_conf"] or 0)))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(DETECTOR)
    cap = cv2.VideoCapture(CLIP)
    print("=== STEP 2a: candidate crops (eyeball these for CLEAN_FISH) ===")
    for tid in tracks:
        recs = [(f, c) for f, c in by_track.get(tid, []) if f not in USED_FRAMES]
        recs.sort(key=lambda x: x[1], reverse=True)
        for f, c in recs[:per_track]:
            p = OUTDIR / f"cand_t{tid}_f{f:04d}.png"
            wh = cut_crop(model, cap, f, c, p)
            if wh:
                print(f"  {p.name}: frame={f} det_conf={c:.3f} crop={wh[0]}x{wh[1]}")
    cap.release()


def write_scores(rows, out_csv):
    cols = ["crop", "scheme", "top1_label", "top1_score", "clarkii_rank",
            "clarkii_score", "clarkii_is_top1", "full_dist"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")
    for r in sorted(rows, key=lambda x: (x["crop"], x["scheme"])):
        print(f'  {r["crop"]:14s} {r["scheme"]}  top1={r["top1_label"][:22]:22s} '
              f'{r["top1_score"]:.3f}  clarkii rank={r["clarkii_rank"]} score={r["clarkii_score"]:.3f}'
              f'{"  <-- clarkii top1" if r["clarkii_is_top1"] else ""}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["extract", "candidates", "score"])
    ap.add_argument("--crops", nargs="*", default=None,
                    help="score phase: explicit list of crop png paths")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.phase == "extract":
        # STEP 1: full distribution on the existing 10 crops.
        existing = sorted(str(p) for p in Path("outputs/bioclip_crops").glob("*.png"))
        print("=== STEP 1: full scores on existing 10 crops (with-clarkii, schemes A+B) ===")
        write_scores(score_crops(existing), OUTDIR / "full_scores_existing.csv")
        print("\nnow run: bioclip_clean_followup.py candidates  (separate process — YOLO and "
              "bioclip can't share one process: open_clip alters torch.load safe-globals)")
    elif args.phase == "candidates":
        # STEP 2a: candidate crops from long-lived tracks for visual confirmation.
        # Must run in its OWN process (no bioclip import) or YOLO weight load fails.
        extract_candidates(tracks=["30", "85", "1", "39", "109", "4"], per_track=3)
    else:
        assert args.crops, "score phase needs --crops <paths>"
        print("=== STEP 2b: full scores on confirmed-clean new crops ===")
        write_scores(score_crops(args.crops), OUTDIR / "full_scores_new_clean.csv")


if __name__ == "__main__":
    main()
