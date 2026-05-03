# Real-Time Fish Species Identification

Real-time fish detection and species identification pipeline combining YOLOv8 object detection with EfficientNet-B0 classification. Trained on a multi-domain dataset spanning 33 fish species across two biogeographic regions.

## Pipeline

```
Video frame (webcam / file)
        ↓
YOLOv8n  ──→  fish bounding boxes
        ↓
EfficientNet-B0 crop classifier  ──→  species + confidence
        ↓
OpenCV overlay  ──→  annotated real-time stream
```

Each frame runs detection first, then per-crop classification only on detected fish — minimising compute and avoiding misclassification on background regions.

## Dataset

| Source | Species | Images | Region |
|---|---|---|---|
| [Fish4Knowledge](https://homepages.inf.ed.ac.uk/rbf/fish4knowledge/) | 23 reef species | ~27,000 | Indo-Pacific |
| iNaturalist (research-grade) | 10 Mediterranean species | ~1,500 | Mediterranean Sea |
| **Combined** | **33 species** | **~28,500** | **Multi-domain** |

**Class imbalance handling**: F4K has ~1,200 images per class vs ~150 for Mediterranean species. A `WeightedRandomSampler` ensures equal class representation during training.

### Mediterranean species
| Species | Common name | iNaturalist observations |
|---|---|---|
| *Sparus aurata* | Gilthead seabream | 2,629 |
| *Diplodus sargus* | White seabream | 6,825 |
| *Scorpaena scrofa* | Red scorpionfish | 1,534 |
| *Mullus surmuletus* | Striped red mullet | 4,511 |
| *Epinephelus marginatus* | Dusky grouper | 4,911 |
| *Coris julis* | Rainbow wrasse | 6,273 |
| *Oblada melanura* | Saddled seabream | 5,065 |
| *Thalassoma pavo* | Ornate wrasse | 8,348 |
| *Sarpa salpa* | Salema porgy | 8,803 |
| *Muraena helena* | Mediterranean moray | 3,567 |

## Model

- **Detector**: YOLOv8n (pretrained on COCO, auto-downloaded)
- **Classifier**: EfficientNet-B0 (ImageNet pretrained, fine-tuned on combined dataset)
- **Training**: Two-stage — head-only warmup → full fine-tune with cosine LR decay, AdamW, early stopping
- **Input size**: 224×224 RGB, ImageNet normalisation
- **Augmentation**: RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomRotation

## Results

| Metric | Value |
|---|---|
| Best validation accuracy | *run to populate* |
| Classes | 33 (23 F4K + 10 Mediterranean) |
| Training images | ~22,800 (80/20 split) |
| Real-time performance | ~15 FPS on CPU / ~60 FPS on GPU |

## Quickstart

```bash
git clone https://github.com/alvaropenuelas/real-time-species-identification
cd real-time-species-identification
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Clone Fish4Knowledge dataset
git clone https://github.com/Callmewuxin/fish4konwledge data/fish4knowledge

# 2. Download Mediterranean species from iNaturalist
python download_med_species.py --per-species 150

# 3. Build combined dataset (symlinks — no disk duplication)
python src/prepare_data.py

# 4. Train — head warmup first
python main.py --feature-extract --epochs 10

# 5. Full fine-tune
python main.py --epochs 25 --lr 3e-4

# 6. Run real-time on webcam
python src/realtime.py --source 0

# 7. Run on a video file and save output
python src/realtime.py --source video.mp4 --save outputs/annotated.mp4
```

## Project structure

```
real-time-species-identification/
├── main.py                    # Training entry point
├── download_med_species.py    # iNaturalist data fetcher
├── assets/                    # Figures (tracked)
├── src/
│   ├── species_map.py         # Species index ↔ display name mapping
│   ├── dataset.py             # Combined DataLoader with WeightedRandomSampler
│   ├── model.py               # EfficientNet-B0 with replaceable head
│   ├── train.py               # Training loop, early stopping, checkpointing
│   ├── prepare_data.py        # Merges F4K + Mediterranean into data/combined/
│   └── realtime.py            # Real-time inference loop (YOLO + classifier)
├── data/                      # ⬇ local only (gitignored)
│   ├── fish4knowledge/        # Cloned from GitHub
│   ├── mediterranean/         # Downloaded from iNaturalist
│   └── combined/              # Symlinked merged dataset
└── outputs/                   # ⬇ local only (gitignored)
    ├── best_model.pt
    ├── classes.json
    └── history.json
```

## Limitations and next steps

- **Detection generalisation**: YOLOv8n is not fine-tuned on fish; it uses general COCO weights. A fish-specific detector (e.g. trained on FathomNet) would improve recall in low-contrast underwater footage.
- **Domain gap**: F4K images are from fixed underwater cameras in Taiwan; Mediterranean images come from citizen science (variable quality, angles, lighting). A domain adaptation stage would improve cross-domain robustness.
- **Towards VLMs**: The classifier requires per-species retraining for new taxa. The natural extension is zero-shot species identification with vision-language models (BioCLIP, CLIP) — eliminating the need for labelled data per new species, which is the critical bottleneck in open-world biodiversity monitoring.

## Stack

Python 3.10+ · PyTorch 2.10 · torchvision · Ultralytics YOLOv8 · OpenCV · scikit-learn · iNaturalist API v1 · Fish4Knowledge
