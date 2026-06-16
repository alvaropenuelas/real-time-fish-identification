"""
Training entry point.

Usage:
    python main.py                        # full fine-tune, 25 epochs
    python main.py --feature-extract      # head-only warmup (run first)
    python main.py --epochs 15 --lr 3e-4  # full fine-tune after warmup
"""

import argparse
import json
import os

import torch

from src.dataset import get_dataloaders
from src.model import build_model, count_trainable_params
from src.train import train

DATA_DIR = "data/combined"
OUTPUT_DIR = "outputs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_DIR)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--feature-extract", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, class_names = get_dataloaders(
        args.data, val_split=args.val_split, batch_size=args.batch
    )
    print(
        f"Classes: {len(class_names)}  |  Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}"
    )

    os.makedirs(args.output, exist_ok=True)
    with open(f"{args.output}/classes.json", "w") as f:
        json.dump(class_names, f, indent=2)

    model = build_model(num_classes=len(class_names), feature_extract=args.feature_extract)
    print(f"Trainable params: {count_trainable_params(model):,}")

    best_acc = train(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        output_dir=args.output,
        device=device,
        lr=args.lr,
        patience=args.patience,
    )
    print(f"\nDone. Best val_acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
