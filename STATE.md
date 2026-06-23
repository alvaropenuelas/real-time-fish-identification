# Project State — Marine Fish Detector

## Current notebook
`notebooks/train_yolo_kaggle.ipynb` — clip-disjoint DeepFish detector (commit `f5b1cbf`).
Fine-tunes `yolov8n` on DeepFish, trained on Kaggle T4.

## Status
Training **running on Kaggle T4** (pushed/run from the UI, not via API — API push lands
on P100). Awaiting the honest before/after mAP from the held-out clip-disjoint TEST.

## Data split — DO NOT REVERT
Split is **clip-disjoint** (whole clips partitioned into train/val/test, seed 42), built
from the annotated DeepFish full export (`FULL_ID`, ~1.1 GB).

**DO NOT revert to the v2 / leaky frame-level split.** Its reported mAP50 ≈ **0.97 was
inflated by frame leakage** — near-identical neighbouring frames from the same clip
straddled train/test, so the model was scored on frames almost identical to ones it
trained on. The clip-disjoint number will be lower, but it is the honest generalization
estimate.

Root cause that forced this: all box-labeled DeepFish frames live in the same 46 clips as
the official frame-level test set, so a clip-disjoint split cannot keep the official test.
Fix: dropped the official test, re-split all 46 labeled clips clip-disjoint.

## Guardrails (keep)
- Keep clip-disjoint split.
- Keep `boxes>0` + clip-overlap asserts (HARD GATE 1 + 2 in cell A2).
- Never revert to the leaky frame-level split.
- Never write empty/placeholder labels for box frames (NOAA backgrounds are the only
  intentionally-empty labels, TRAIN-only, capped ≤10%).

## Done
- Clip-disjoint train/val/test split from full export, with leak + pairing asserts.
- Fixed empty-train-labels bug (annotation lookup against full export).
- Pinned torch/torchvision so Kaggle pip install keeps the GPU-matched build.
- NOAA "Labeled Fishes in the Wild" negatives as hard-negative backgrounds (TRAIN only).
- BEFORE/AFTER eval scaffold: freshwater detector vs new yolov8n on same held-out TEST.

## Next step
Run clip-disjoint training on Kaggle T4 (from UI). Paste before/after mAP back in.
Record the honest clip-disjoint TEST mAP50 / mAP50-95 and the before→after delta.
