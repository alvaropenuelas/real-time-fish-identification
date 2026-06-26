"""BioCLIP 2 zero-shot test on fish_clip_2 crops vs the closed-world EfficientNet.

Isolated experiment. Does NOT import or modify realtime.py or the EfficientNet classifier.
Ground truth for the whole clip: Amphiprion clarkii (Clark's anemonefish).

Pipeline:
  STEP 1  pick easy/hard crops deterministically from the prior EfficientNet predict-log
          CSV (by track duration + detector confidence — box size was not logged there),
          then re-run the marine detector on those exact frames to cut the crop pixels.
  STEP 3  for every crop x {scheme A scientific, scheme B common} x {with, without clarkii}
          run CustomLabelsClassifier (BioCLIP 2, CPU) and record top1/top2/margin/entropy.

API note (verified against pybioclip 2.1.5, not assumed): CustomLabelsClassifier(cls_ary,
device='cpu'); predict(images, k=None) returns a LIST of dicts, one per image, shaped
{"file_name": <path>, "<label>": <softmax prob>, ...}. Scores are softmax over the given
label list, so dropping clarkii renormalizes over the remaining labels.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2
from PIL import Image

CLIP = "fish_clip_2.mp4"
DETECTOR = "weights/detector_deepfish.pt"
PRED_CSV = "outputs/demo_fish_clip_2_predictions.csv"
CROPS_DIR = Path("outputs/bioclip_crops")
RESULTS_CSV = Path("outputs/bioclip_zeroshot_results.csv")
N_PER_GROUP = 5

# clarkii is index 0 in each scheme; "without" drops exactly that label.
SCHEME_A = ["Amphiprion clarkii", "Amphiprion ocellaris", "Amphiprion percula",
            "Pomacentrus moluccensis", "Dascyllus reticulatus", "Chromis chrysura",
            "Sea anemone", "Coral reef"]
SCHEME_B = ["Clark's anemonefish", "Common clownfish", "Orange clownfish",
            "Lemon damselfish", "Reticulated dascyllus", "Pacific chromis",
            "Sea anemone", "Coral reef"]
CLARKII = {"A": "Amphiprion clarkii", "B": "Clark's anemonefish"}
SCHEMES = {"A": SCHEME_A, "B": SCHEME_B}


def entropy_bits(probs):
    return -sum(p * math.log2(p) for p in probs if p > 0)


def select_crops():
    """Return [(crop_id, difficulty, frame_idx, target_det_conf)] from the CSV, by rule."""
    rows = list(csv.DictReader(open(PRED_CSV)))
    by_track = defaultdict(list)
    for r in rows:
        if r["track_id"]:  # skip untracked (no stable identity)
            by_track[r["track_id"]].append(
                (int(r["frame_idx"]), float(r["det_conf"]) if r["det_conf"] else 0.0)
            )
    stats = {}
    for tid, recs in by_track.items():
        confs = [c for _, c in recs]
        stats[tid] = {
            "n": len(recs), "mean": sum(confs) / len(confs),
            "max_rec": max(recs, key=lambda x: x[1]),   # (frame, conf) best view
            "min_rec": min(recs, key=lambda x: x[1]),   # (frame, conf) worst view
        }
    # EASY: long-lived AND confident. HARD: low mean conf AND short-lived (disjoint).
    easy = sorted(stats.items(), key=lambda kv: (kv[1]["n"], kv[1]["mean"]), reverse=True)[:N_PER_GROUP]
    easy_ids = {tid for tid, _ in easy}
    hard = sorted(((t, s) for t, s in stats.items() if t not in easy_ids),
                  key=lambda kv: (kv[1]["mean"], kv[1]["n"]))[:N_PER_GROUP]

    sel = []
    for i, (tid, s) in enumerate(easy):
        f, c = s["max_rec"]
        sel.append((f"easy_{i:02d}", "easy", f, c, tid, s["n"]))
    for i, (tid, s) in enumerate(hard):
        f, c = s["min_rec"]
        sel.append((f"hard_{i:02d}", "hard", f, c, tid, s["n"]))
    return sel


def extract_crops(selection):
    """Re-run the detector on each selected frame; crop the box whose conf matches the CSV."""
    from ultralytics import YOLO
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(DETECTOR)
    cap = cv2.VideoCapture(CLIP)
    meta = []
    print("=== STEP 1: crop selection + extraction ===")
    print(f'{"crop_id":9s} {"diff":5s} {"track":6s} {"n_frm":>5s} {"frame":>5s} '
          f'{"csv_conf":>8s} {"matched":>7s} {"crop_wxh":>10s}')
    for crop_id, diff, frame_idx, target_conf, tid, n in selection:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"  WARN cannot read frame {frame_idx} for {crop_id}")
            continue
        boxes = model.predict(frame, conf=0.1, imgsz=640, verbose=False)[0].boxes
        if len(boxes) == 0:
            print(f"  WARN no boxes on frame {frame_idx} for {crop_id}")
            continue
        best = min(boxes, key=lambda b: abs(float(b.conf[0]) - target_conf))
        mc = float(best.conf[0])
        x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame[y1:y2, x1:x2]
        path = CROPS_DIR / f"{crop_id}.png"
        Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(path)
        meta.append((crop_id, diff, str(path)))
        print(f'{crop_id:9s} {diff:5s} {tid:6s} {n:5d} {frame_idx:5d} '
              f'{target_conf:8.3f} {mc:7.3f} {f"{x2-x1}x{y2-y1}":>10s}')
    cap.release()
    return meta


def run_bioclip(meta):
    import torch

    # pybioclip calls torch.compile() unconditionally in load_pretrained_model, but torch
    # 2.2's Dynamo is unsupported on Python 3.12 (RuntimeError: Dynamo is not supported on
    # Python 3.12+). torch.compile is only a speed optimization, so fall back to eager.
    torch.compile = lambda model=None, *a, **k: model

    from bioclip import CustomLabelsClassifier
    results = []
    paths = [p for _, _, p in meta]
    diff_of = {p: d for _, d, p in meta}
    id_of = {p: cid for cid, _, p in meta}
    first = True
    for scheme_key, labels in SCHEMES.items():
        for cond in ("with", "without"):
            cls = labels if cond == "with" else [x for x in labels if x != CLARKII[scheme_key]]
            clf = CustomLabelsClassifier(cls_ary=cls, device="cpu")
            if first:
                print(f"\n=== STEP 3: BioCLIP 2 zero-shot | device = {clf.device} ===")
                first = False
            # predict() returns a FLAT list of {file_name, classification, score} rows,
            # one per image x label (verified against pybioclip 2.1.5) — group by image.
            preds = clf.predict(paths, k=None)
            by_file = defaultdict(list)
            for prow in preds:
                by_file[prow["file_name"]].append((prow["classification"], float(prow["score"])))
            for fn, scores in by_file.items():
                scores.sort(key=lambda x: x[1], reverse=True)
                (t1l, t1s), (t2l, t2s) = scores[0], scores[1]
                results.append({
                    "crop_id": id_of[fn], "difficulty": diff_of[fn],
                    "scheme": scheme_key, "condition": cond,
                    "top1_label": t1l, "top1_score": round(t1s, 4),
                    "top2_label": t2l, "top2_score": round(t2s, 4),
                    "margin": round(t1s - t2s, 4),
                    "entropy_bits": round(entropy_bits([s for _, s in scores]), 4),
                })
    return results


def report(results):
    cols = ["crop_id", "difficulty", "scheme", "condition", "top1_label", "top1_score",
            "top2_label", "top2_score", "margin", "entropy_bits"]
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)

    print("\n=== STEP 4: full results ===")
    print(f'{"crop":8s} {"diff":4s} {"sch":3s} {"cond":7s} {"top1":24s} {"t1":>5s} '
          f'{"top2":20s} {"t2":>5s} {"marg":>5s} {"H":>5s}')
    for r in sorted(results, key=lambda x: (x["crop_id"], x["scheme"], x["condition"])):
        print(f'{r["crop_id"]:8s} {r["difficulty"]:4s} {r["scheme"]:3s} {r["condition"]:7s} '
              f'{r["top1_label"][:24]:24s} {r["top1_score"]:5.2f} {r["top2_label"][:20]:20s} '
              f'{r["top2_score"]:5.2f} {r["margin"]:5.2f} {r["entropy_bits"]:5.2f}')

    clk = {"A": CLARKII["A"], "B": CLARKII["B"]}
    print("\n=== SUMMARY (a) WITH clarkii: how often is clarkii top-1 ===")
    for diff in ("easy", "hard"):
        for sk in ("A", "B"):
            sub = [r for r in results if r["condition"] == "with"
                   and r["difficulty"] == diff and r["scheme"] == sk]
            hits = [r for r in sub if r["top1_label"] == clk[sk]]
            if sub:
                avg_s = sum(r["top1_score"] for r in hits) / len(hits) if hits else 0.0
                avg_m = sum(r["margin"] for r in hits) / len(hits) if hits else 0.0
                print(f"  {diff:4s} scheme {sk}: clarkii top1 {len(hits)}/{len(sub)} "
                      f"| avg score(when hit)={avg_s:.3f} avg margin={avg_m:.3f}")

    print("\n=== SUMMARY (b) WITHOUT clarkii: confident-wrong vs honest-uncertain ===")
    for sk in ("A", "B"):
        for diff in ("easy", "hard"):
            sub = [r for r in results if r["condition"] == "without"
                   and r["difficulty"] == diff and r["scheme"] == sk]
            if sub:
                am = sum(r["margin"] for r in sub) / len(sub)
                ah = sum(r["entropy_bits"] for r in sub) / len(sub)
                asc = sum(r["top1_score"] for r in sub) / len(sub)
                print(f"  scheme {sk} {diff:4s}: avg top1={asc:.3f} avg margin={am:.3f} "
                      f"avg entropy={ah:.3f} bits  (low margin+high entropy = honest uncertainty)")
    print(f"\nwrote {RESULTS_CSV}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extract", action="store_true",
                    help="reuse existing crops in outputs/bioclip_crops/ if present")
    args = ap.parse_args()

    for need in (CLIP, DETECTOR, PRED_CSV):
        if not Path(need).exists():
            raise SystemExit(f"missing required input: {need}")

    selection = select_crops()
    existing = sorted(CROPS_DIR.glob("*.png"))
    if args.skip_extract and len(existing) >= 2 * N_PER_GROUP:
        print(f"reusing {len(existing)} existing crops in {CROPS_DIR}")
        meta = [(p.stem, "easy" if p.stem.startswith("easy") else "hard", str(p)) for p in existing]
    else:
        meta = extract_crops(selection)
    if len(meta) < 2 * N_PER_GROUP:
        raise SystemExit(f"only {len(meta)} crops extracted; need {2 * N_PER_GROUP}")

    results = run_bioclip(meta)
    report(results)


if __name__ == "__main__":
    main()
