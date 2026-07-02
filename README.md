# Real-Time Fish Species Identification

A two-stage computer vision pipeline for marine fish: a fine-tuned **YOLOv8n** detector finds fish
in each video frame, **ByteTrack** links detections into per-individual tracks, and a fine-tuned
**EfficientNet-B0** classifier assigns each crop to one of 33 species. It runs on webcam or video
input and annotates the stream with species name and top-3 confidence.

This README doubles as an honest lab notebook. Beyond the "how to run it" material, it consolidates
three closed investigations — into the **detector**, the **closed-world classifier**, and an
**open-vocabulary classifier** — each with what was tried, what was measured, what it means, and
what its limitation is. The recurring finding is that the bottleneck is *domain coverage of the
training/reference data*, not model capacity. Numbers here are reported as measured, including the
ones that are unflattering; where a result is still pending a run it is left as an explicit
placeholder rather than filled with a guess.

## Architecture

The pipeline is two-stage: YOLOv8n runs per-frame to produce fish bounding boxes, each box is
tracked across frames with ByteTrack, and each crop is passed independently to EfficientNet-B0 for
species classification. This avoids running the classifier on background regions and makes the two
models independently replaceable. Both models were trained on Kaggle free GPU (NVIDIA T4).

```
Video frame
      ↓
YOLOv8n  ──→  fish bounding boxes
      ↓
ByteTrack  ──→  per-individual track ids
      ↓  (per crop, skip if < 32×32 px)
EfficientNet-B0  ──→  species + top-3 confidence
      ↓
Annotator  ──→  green bbox + label pill + secondary predictions
```

## Project structure

```
real-time-species-identification/
├── src/
│   ├── realtime.py        # Main inference loop (YOLO + ByteTrack + classifier + annotator)
│   ├── classifier.py      # FishClassifier: loads EfficientNet, returns top-3 predictions
│   ├── annotator.py       # Draws bounding boxes and label pills on frames
│   ├── prepare_data.py    # Merges Fish4Knowledge + Mediterranean into data/combined/
│   ├── train.py           # EfficientNet training loop
│   ├── model.py           # EfficientNet-B0 with fine-tuned classification head
│   ├── dataset.py         # DataLoader with WeightedRandomSampler
│   └── species_map.py     # Folder name → display name mapping
├── notebooks/
│   ├── train_kaggle.ipynb      # EfficientNet classifier training (Kaggle GPU)
│   └── train_yolo_kaggle.ipynb # YOLOv8 detector training + clip-disjoint eval (Kaggle GPU)
├── tests/
│   └── test_classifier.py      # Smoke test for FishClassifier
├── weights/                    # Not in repo — see Setup
│   ├── model.pt           # EfficientNet-B0 classifier weights
│   └── yolo_fish.pt       # YOLOv8n fish detector weights
└── outputs/
    ├── classes.json       # 33-class label list (alphabetical)
    ├── metrics.json       # Classifier eval + latency
    ├── preproc_test/      # Detector ghost-box diagnosis + conf sweep
    └── bioclip_clean/     # BioCLIP-2 zero-shot investigation
```

## Setup

```bash
git clone https://github.com/alvaropenuelas/real-time-species-identification
cd real-time-species-identification
pip install -r requirements.txt
```

Weights are not tracked in the repository. Train them using the Kaggle notebooks (see Training
below) or obtain pre-trained checkpoints:

- Place EfficientNet classifier weights at `weights/model.pt`
- Place YOLOv8 detector weights at `weights/yolo_fish.pt`

## Usage

```bash
# Video file
python src/realtime.py --source path/to/video.mp4

# Video file with FPS printed to terminal every 30 frames
python src/realtime.py --source path/to/video.mp4 --display-fps
```

Press `q` to quit.

Additional options:

| Flag | Default | Description |
|---|---|---|
| `--weights` | `weights/model.pt` | Path to EfficientNet classifier weights |
| `--detector` | `weights/yolo_fish.pt` | Path to YOLOv8 detector weights |
| `--conf` | `0.5` | Minimum classifier confidence threshold |

## Training

Both models were trained on Kaggle free GPU (NVIDIA T4) using notebooks in `notebooks/`:

- **Classifier** (`train_kaggle.ipynb`): EfficientNet-B0 fine-tuned on Fish4Knowledge + iNaturalist
  Mediterranean species (33 classes, ~4,700 images after subsampling). Data uploaded to Kaggle as
  dataset `lvarop99/fish-training-data`.
- **Detector** (`train_yolo_kaggle.ipynb`): YOLOv8n fine-tuned on **DeepFish** (marine underwater
  video, single-class fish), using a clip-disjoint split (see below).

---

# Findings

Three investigations, in pipeline order. Each is closed; each is honest about its limitation.

## 1. Detector (YOLOv8n on DeepFish)

### What was tried, and a data leak

The detector was retrained from a freshwater dataset onto **DeepFish** (marine underwater video,
from the YOLO-Fish benchmark) to close a training/deployment domain mismatch. The first attempt
used a **frame-level split** — individual annotated frames assigned to train/val/test
independently — and scored **mAP@0.5 ≈ 0.97**. That was a red flag, not a success. DeepFish frames
come from a small number of video clips, and consecutive frames within a clip are near-duplicates.
A frame-level split scatters those near-duplicates across train and test, so the model was scored
partly on frames it had effectively already seen. Inspecting train vs test frames confirmed the
leak directly. The 0.97 measured memorisation, not generalisation.

### The fix — a clip-disjoint split

The split was rebuilt so **every video clip lives in exactly one of train/val/test**. Frames are
grouped by clip id (filename with the trailing `_f######` frame index stripped) and whole clips —
not individual frames — are partitioned (deterministic, seed 42). The test set then contains only
clips never seen in training, so it measures generalisation to unseen footage.

| Split | Clips | Images | Boxes |
|---|---:|---:|---:|
| train | 31 | 2986 | 11258 |
| val   | 6  | 392  | 695   |
| test  | 9  | 1127 | 3510  |

> Note: these per-split counts are computed at runtime inside `train_yolo_kaggle.ipynb` from the
> DeepFish export (not committed), and have not been reproduced against a completed run's printed
> output. The split is deterministic (seed 42) so they should be stable. The **test** figures
> (1127 imgs / 3510 boxes) are independently corroborated by the ghost-box report; treat the
> train/val figures as expected-but-unverified.

Two permanent runtime assertions guard the two failure modes so they cannot silently return:
**no empty labels** (every label file in each split has ≥1 box, else the run aborts) and
**clip-disjoint** (train/val/test clip sets must be pairwise disjoint, else the run aborts). Both
run every time the dataset is assembled.

### What was measured — honest before/after

Same held-out clip-disjoint test set (1127 imgs / 3510 boxes), same eval settings for both models:

| Detector | Training domain | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---:|---:|---:|---:|
| Freshwater (old) | out-of-domain | 0.022 | 0.012 | 0.068 | 0.056 |
| DeepFish (new)   | in-domain     | 0.593 | 0.316 | 0.764 | 0.560 |

(New detector weights: sha256 starting `6688454003028a6c`.)

### What it means

This is an **in-domain (marine-trained) vs out-of-domain (freshwater-trained)** comparison on the
same test set. It shows domain match matters for this task — it is **not** a claim of
state-of-the-art detection. The official YOLO-Fish benchmark test split was deliberately dropped:
its frames share clips with the training data, so using it would reintroduce exactly the
clip-overlap leak above. This result is therefore intentionally **not comparable** to the published
benchmark numbers (~0.76–0.92), which are themselves very likely in-distribution with frame-level
splits.

### Limitations

- **Recall.** R = 0.56 means roughly **44% of fish are missed per frame**. Video deployment runs
  ByteTrack, so per-individual recall across a whole track is *likely* higher than per-frame
  recall — but this has **not been measured**; it is an open question for the pending video test
  set, not a fix.
- **Reef clutter / false positives (known, unresolved).** On high-resolution reef footage (the
  4K Pexels demo clip) the detector fires on background structure — anemone tentacles, coral,
  water particles. This was checked and is **not** an aspect-ratio/preprocessing bug: squashing to
  640² *reduced* box count (28 → 12), it did not inflate it. Diagnosis: a **domain gap** —
  DeepFish's open seagrass/marine distribution doesn't cover reef-aquarium clutter. A confidence
  sweep shows the trade-off qualitatively — ghost boxes drop from ~3.3/frame at conf 0.25 to
  ~0.3/frame at conf 0.8, but by 0.8 the two ground-truth *clarkii* are also lost. The quantitative
  per-threshold test-set mAP/recall table is `{{conf_sweep_test_table}}` (pending a Kaggle run,
  since the test split isn't on the local machine). The durable fix is **hard-negative background
  training data**, not just a higher threshold; this is logged as a known open limitation, not
  resolved.

### A genuine strength — true negatives

On **147 NOAA no-fish frames**, the detector produced **1 total false-positive box**. Small sample,
but real, and most portfolio projects never test true negatives at all.

## 2. Closed-world classifier (EfficientNet-B0, 33 species)

### What was measured — baseline

In-domain, on posed/reference photos: **top-1 accuracy 0.94, macro-F1 0.947** (n_val = 783,
33 classes; see `outputs/metrics.json`).

### Deployment test on real-world footage

The full pipeline (marine detector + ByteTrack + EfficientNet, with the `HARD_CONF_FLOOR` /
threshold logic and default `--conf 0.5`) was run on a real-world Pexels clip containing
*Amphiprion clarkii* (Clark's anemonefish) — a species **in** the classifier's known set,
confirmed by visual inspection rather than detector confidence alone.

- **872** classified detections; **187** emitted (≥ 0.5 confidence); **685** suppressed.
- *Clarkii* correct in **0 / 187 emitted predictions (0.0%)**.
- Wrong labels were frequently **high-confidence** — e.g. *Pomacentrus moluccensis* 0.93,
  Saddled seabream 0.85 — not low-confidence noise.

### What it means

Having the correct species in the label set is **necessary but not sufficient**. The classifier was
trained on Fish4Knowledge's fixed-camera reef stills and fails to generalise to different footage
conditions (lighting, compression, camera motion) even for a species it nominally knows. This is
the **closed-world failure mode**: the model cannot say "I don't know" — it always forces one of 33
labels and attaches a confidence score, so it is *confidently wrong* rather than uncertain.

### Performance / limitation

Real-world throughput on this clip: **2.75 FPS wall-clock** at native **2160×3840, CPU, no GPU**.
That is far from real-time at native 4K on CPU — resolution and hardware context matter when reading
the "real-time" in the project name. (A separate CPU latency probe on a smaller 120-frame clip
recorded ~3.7 end-to-end FPS; both are CPU-only and not indicative of the GPU target.)

## 3. Open-vocabulary classifier (BioCLIP-2, zero-shot)

### Motivation

Test whether a foundation-model **zero-shot** classifier avoids the closed-world failure above,
without committing to a full architecture rewrite.

### Method

`CustomLabelsClassifier` (pybioclip, BioCLIP-2 default weights) on crops from the same Pexels clip.
Two label-phrasing schemes were tested separately — **scheme A: scientific names**, **scheme B:
common names** — because label wording is documented to bias BioCLIP's output. Each was run with and
without the true label (*clarkii*) present in the candidate list.

### A correction mid-investigation

The initial crop selection (by detector confidence / track duration) turned out to include
**6 of 10 crops that were background-dominant detector false positives** (anemone/coral, no real
fish) — consistent with Section 1's domain-gap finding. Results were re-cut by **visual
confirmation**, and **6 additional crops were newly sampled and visually verified as clean fish**
before testing.

### What was measured — final result

On **n = 6 independent, visually-confirmed clean fish crops** (12 runs across 2 schemes):
*clarkii* was **top-1 in 0 / 12**.

- **Scheme A (scientific names)** collapses hard onto *Amphiprion percula* (high margin) on 5 of 6
  clean crops.
- **Scheme B (common names)** is more dispersed; *clarkii*'s best result on a clean crop was
  **rank #2 at score 0.246** — but never top-1.

One single *clarkii* top-1 hit (score **0.978**) occurred earlier on a **mixed** crop showing the
clownfish sitting inside its **host anemone**. Whether the surrounding anemone context cued the
correct anemonefish label is a **speculative, unpursued hypothesis (n = 1)** — flagged, not a
validated finding.

### What it means

**BioCLIP-2 zero-shot does not outperform the closed-world EfficientNet** at fine-grained
*Amphiprion* species identification on this clip. Like EfficientNet, it is **confidently wrong, not
uncertain** — even on clean crops.

### A genuine (narrow) advantage

On the background-dominant false-positive crops — the detector's ghost boxes — BioCLIP's open
vocabulary correctly returned **"Sea anemone"** instead of forcing a fish label, something
EfficientNet structurally cannot do. This is a real but **narrow** advantage — background
rejection, not a species-ID win.

## Conclusion

All three investigations point to the same bottleneck: **domain coverage of the training/reference
data, not model capacity or architecture generation.** The detector's ghosts, the closed-world
classifier's confident mislabels, and BioCLIP's zero-shot failures all trace back to a mismatch
between the data the models learned from and the reef footage they were tested on — swapping in a
newer model did not close that gap. The next planned step is therefore a **real underwater video
test set** (everything measured so far is posed photos or a single out-of-domain clip), not another
model swap. This is an honest diagnostic portfolio piece, not a finished product.

## Author

Alvaro Peñuelas Suria — MSc Marine and Lacustrine Science and Management (Ghent/VUB/Antwerp),
BSc Biology (Universidad de Navarra). GitHub: [alvaropenuelas](https://github.com/alvaropenuelas)

This project was developed independently as part of a portfolio demonstrating real-time computer
vision applied to marine biology. AI tools (Claude Code) were used as a coding assistant; all
architectural decisions, dataset curation, evaluation, and project direction are my own.

## Citations and data sources

### Datasets

- **Fish4Knowledge (F4K)** — Underwater video dataset for fish detection and recognition. Used for
  23 species classes in the classifier.
  Boom, B.J., Huang, P.X., He, J., Fisher, R.B. (2012). "Supporting ground-truth annotation of
  image datasets using clustering." Pattern Recognition, ICPR 2012.

- **iNaturalist** — Citizen science observations of Mediterranean fish species. Used for 10
  Mediterranean species classes in the classifier. iNaturalist API. https://www.inaturalist.org

- **DeepFish / YOLO-Fish** — Marine underwater video, single-class fish detection, used to train
  the YOLOv8 detector (clip-disjoint split).
  Saleh, A. et al. (2020). "A realistic fish-habitat dataset to evaluate algorithms for underwater
  visual analysis." Scientific Reports.

- **NOAA "Labeled Fishes in the Wild"** — Used as hard-negative background frames (train only).

### Models and frameworks

- **YOLOv8 / ByteTrack** — Ultralytics (2023). https://github.com/ultralytics/ultralytics
- **EfficientNet** — Tan, M., Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for
  Convolutional Neural Networks." ICML 2019.
- **BioCLIP-2 / pybioclip** — Foundation model for the tree of life, used zero-shot.
- **PyTorch** — Paszke, A. et al. (2019). NeurIPS 2019.
- **OpenCV** — Bradski, G. (2000). Dr. Dobb's Journal of Software Tools.

### Computational resources

- **Kaggle** — Free GPU (NVIDIA T4) used for training both models. https://www.kaggle.com

## License

MIT
