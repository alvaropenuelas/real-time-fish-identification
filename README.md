# Real-Time Fish Species Identification

A real-time computer vision pipeline that detects fish in video frames using a fine-tuned YOLOv8n detector, then classifies each detected fish into one of 33 species using a fine-tuned EfficientNet-B0 classifier. The system runs on webcam or video file input and annotates the output stream with species name and top-3 confidence predictions.

## Architecture

The pipeline is two-stage: YOLOv8n runs per-frame to produce fish bounding boxes, then each crop is passed independently to EfficientNet-B0 for species classification. This avoids running the classifier on background regions and makes the two models independently replaceable. Both models were trained on Kaggle free GPU (NVIDIA T4).

```
Video frame
      ↓
YOLOv8n  ──→  fish bounding boxes
      ↓  (per crop, skip if < 32×32 px)
EfficientNet-B0  ──→  species + top-3 confidence
      ↓
Annotator  ──→  green bbox + label pill + secondary predictions
```

## Project structure

```
real-time-species-identification/
├── src/
│   ├── realtime.py        # Main inference loop (YOLO + classifier + annotator)
│   ├── classifier.py      # FishClassifier: loads EfficientNet, returns top-3 predictions
│   ├── annotator.py       # Draws bounding boxes and label pills on frames
│   ├── prepare_data.py    # Merges Fish4Knowledge + Mediterranean into data/combined/
│   ├── train.py           # EfficientNet training loop
│   ├── model.py           # EfficientNet-B0 with fine-tuned classification head
│   ├── dataset.py         # DataLoader with WeightedRandomSampler
│   └── species_map.py     # Folder name → display name mapping
├── notebooks/
│   ├── train_kaggle.ipynb      # EfficientNet classifier training (Kaggle GPU)
│   └── train_yolo_kaggle.ipynb # YOLOv8 detector training (Kaggle GPU)
├── tests/
│   └── test_classifier.py      # Smoke test for FishClassifier
├── weights/                    # Not in repo — see Setup
│   ├── model.pt           # EfficientNet-B0 classifier weights
│   └── yolo_fish.pt       # YOLOv8n fish detector weights
└── outputs/
    └── classes.json       # 33-class label list (alphabetical)
```

## Setup

```bash
git clone https://github.com/alvaropenuelas/real-time-species-identification
cd real-time-species-identification
pip install -r requirements.txt
```

Weights are not tracked in the repository. Train them using the Kaggle notebooks (see Training below) or obtain pre-trained checkpoints:

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

- **Classifier** (`train_kaggle.ipynb`): EfficientNet-B0 fine-tuned on Fish4Knowledge + iNaturalist Mediterranean species (33 classes, ~4,700 images after subsampling). Data uploaded to Kaggle as dataset `lvarop99/fish-training-data`.
- **Detector** (`train_yolo_kaggle.ipynb`): YOLOv8n fine-tuned on a single-class fish bounding box dataset from Roboflow Universe (50 epochs, imgsz=640). Data uploaded as `lvarop99/fish-dataset-v1`.

## Limitations

- Classifier trained on ~4,700 images — limited generalisation to diverse video conditions
- Detector trained on a freshwater fish dataset — generalises partially to marine species
- Best results on clear underwater footage with a single fish centred in frame

## Author

Alvaro Peñuelas Suria — MSc Marine and Lacustrine Science and Management (Ghent/VUB/Antwerp), BSc Biology (Universidad de Navarra). GitHub: [alvaropenuelas](https://github.com/alvaropenuelas)

This project was developed independently as part of a portfolio demonstrating real-time computer vision applied to marine biology. AI tools (Claude Code) were used as a coding assistant; all architectural decisions, dataset curation, evaluation, and project direction are my own.

## Citations and data sources

### Datasets

- **Fish4Knowledge (F4K)** — Underwater video dataset for fish detection and recognition. Used for 23 species classes in the classifier.
  Boom, B.J., Huang, P.X., He, J., Fisher, R.B. (2012). "Supporting ground-truth annotation of image datasets using clustering." Pattern Recognition, ICPR 2012.

- **iNaturalist** — Citizen science observations of Mediterranean fish species. Used for 10 Mediterranean species classes in the classifier.
  iNaturalist API. https://www.inaturalist.org

- **Roboflow Universe — Fish Detection Dataset** — Single-class fish bounding box dataset used to train the YOLOv8 detector.
  YOLO. (2024). Fish Dataset (v1). Roboflow Universe. https://universe.roboflow.com/yolo-zbpxw/fish-ku7kf
  License: CC BY 4.0

### Models and frameworks

- **YOLOv8** — Ultralytics (2023). YOLOv8 by Ultralytics. https://github.com/ultralytics/ultralytics
- **EfficientNet** — Tan, M., Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." ICML 2019.
- **PyTorch** — Paszke, A. et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." NeurIPS 2019.
- **OpenCV** — Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.

### Computational resources

- **Kaggle** — Free GPU (NVIDIA T4) used for training both the EfficientNet classifier and YOLOv8 detector. https://www.kaggle.com

## License

MIT
