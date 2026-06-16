"""
Evaluate the current EfficientNet classifier on its held-out val split.

Reuses src.dataset.get_dataloaders so the val set is the SAME seed-42 split
(random_split, generator manual_seed=42) the training run held out. Reports
top-1 accuracy, macro-F1, and writes a confusion-matrix PNG. Numbers are
appended to outputs/metrics.json under phase0_baseline.classifier_eval.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import get_dataloaders
from src.model import build_model
from src.species_map import DISPLAY_NAMES


def main():
    parser = argparse.ArgumentParser(description="Eval classifier on held-out val split")
    parser.add_argument("--data-dir", default="data/combined")
    parser.add_argument("--weights", default="weights/model.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--metrics-path", default="outputs/metrics.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same seed-42 val split the training used; num_workers=0 for determinism/macOS.
    _, val_loader, classes = get_dataloaders(
        args.data_dir, val_split=0.2, batch_size=args.batch_size, num_workers=0
    )
    num_classes = len(classes)

    state_dict = torch.load(args.weights, map_location=device, weights_only=True)
    model = build_model(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    top1 = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    display = [DISPLAY_NAMES.get(c, c.replace("_", " ")) for c in classes]
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    fig_w = max(8, num_classes * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(display, rotation=90, fontsize=6)
    ax.set_yticklabels(display, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix — top1={top1:.3f} macroF1={macro_f1:.3f} (n={len(all_labels)})")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    metrics = {
        "n_val": len(all_labels),
        "num_classes": num_classes,
        "top1_accuracy": round(top1, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix_png": str(cm_path),
    }

    mp = Path(args.metrics_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if mp.exists():
        try:
            existing = json.loads(mp.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing.setdefault("phase0_baseline", {})["classifier_eval"] = metrics
    mp.write_text(json.dumps(existing, indent=2))

    print("=== CLASSIFIER EVAL (held-out val split, seed 42) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
