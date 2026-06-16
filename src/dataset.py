"""
Combined classifier DataLoader (Fish4Knowledge + Mediterranean / MEDFISH101), plus the
detector augmentation pipeline used for YOLO fine-tuning on Kaggle.

Augmentation is Albumentations and applies at TRAINING time only — underwater video needs
contrast/colour/blur/noise robustness (CLAHE, brightness/contrast, hue/sat, motion blur,
Gaussian noise). The VAL pipeline and the live inference path (src/classifier.py) stay
clean — NO CLAHE, NO augmentation — so eval and deployment see undistorted frames.

NOTE: Albumentations + opencv-python-headless install from wheels on Linux (Kaggle/CI/
Docker). On this macOS x86_64 dev box they would source-build, so albumentations is not
imported by the always-on inference path — only by training (Kaggle) and eval tooling.
"""

from collections import Counter
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_train_transform() -> A.Compose:
    """TRAINING-TIME classifier augmentation. CLAHE etc. live here and ONLY here."""
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=256),
            A.RandomCrop(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.4),
            A.MotionBlur(blur_limit=7, p=0.3),
            A.GaussNoise(p=0.3),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )


def build_val_transform() -> A.Compose:
    """Clean eval preprocessing — matches the inference path, NO augmentation/CLAHE."""
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=256),
            A.CenterCrop(height=224, width=224),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )


def build_detector_train_transform() -> A.Compose:
    """YOLO detector fine-tuning augmentation (Kaggle). Same photometric augs, with
    bounding boxes tracked in YOLO format (cx, cy, w, h normalized)."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.4),
            A.MotionBlur(blur_limit=7, p=0.3),
            A.GaussNoise(p=0.3),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


class _AlbTransform:
    """Adapt an Albumentations Compose to torchvision ImageFolder (PIL in, tensor out)."""

    def __init__(self, compose: A.Compose):
        self.compose = compose

    def __call__(self, pil_img):
        return self.compose(image=np.array(pil_img))["image"]


def _restrict_to_top_n(ds: datasets.ImageFolder, top_n: int) -> datasets.ImageFolder:
    """Keep only the top_n most-populated classes, relabelled to a contiguous 0..top_n-1.
    Deterministic (frequency then name) so train/val ImageFolders filter identically."""
    counts = Counter(label for _, label in ds.samples)
    keep_ids = {cls for cls, _ in counts.most_common(top_n)}
    keep_names = sorted(ds.classes[c] for c in keep_ids)
    new_idx = {name: i for i, name in enumerate(keep_names)}
    old_to_new = {ds.class_to_idx[name]: new_idx[name] for name in keep_names}
    samples = [(p, old_to_new[label]) for p, label in ds.samples if label in keep_ids]
    ds.samples = ds.imgs = samples
    ds.targets = [label for _, label in samples]
    ds.classes = keep_names
    ds.class_to_idx = new_idx
    return ds


def _make_weighted_sampler(dataset) -> WeightedRandomSampler:
    """Upsample minority classes so each class is seen equally per epoch."""
    class_counts = torch.zeros(len(dataset.classes))
    for _, label in dataset.samples:
        class_counts[label] += 1
    weights = 1.0 / class_counts
    sample_weights = torch.tensor([weights[label] for _, label in dataset.samples])
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def get_dataloaders(
    data_dir: str, val_split: float = 0.2, batch_size: int = 32, num_workers: int = 4, top_n: int = 0
):
    """top_n=0 uses all classes; top_n>0 keeps the N most-populated species
    (the MEDFISH101 'top-N-frequent subset' knob)."""
    data_dir = Path(data_dir)

    full = datasets.ImageFolder(data_dir, transform=_AlbTransform(build_train_transform()))
    if top_n and 0 < top_n < len(full.classes):
        full = _restrict_to_top_n(full, top_n)
    n_val = int(len(full) * val_split)
    n_train = len(full) - n_val
    train_set, val_set = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    # Val subset must use the clean transform — swap the underlying dataset (filtered identically).
    val_base = datasets.ImageFolder(data_dir, transform=_AlbTransform(build_val_transform()))
    if top_n and 0 < top_n < len(val_base.classes):
        val_base = _restrict_to_top_n(val_base, top_n)
    val_set.dataset = val_base

    sampler = _make_weighted_sampler(full)
    # Only apply sampler to train indices
    train_sampler = WeightedRandomSampler(
        [sampler.weights[i] for i in train_set.indices], num_samples=n_train, replacement=True
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, full.classes
