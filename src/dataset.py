"""
Builds a combined DataLoader from Fish4Knowledge + Mediterranean datasets.
Handles class imbalance (F4K ~1200 imgs/class vs Med ~150) via WeightedRandomSampler.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _make_weighted_sampler(dataset) -> WeightedRandomSampler:
    """Upsample minority classes so each class is seen equally per epoch."""
    class_counts = torch.zeros(len(dataset.classes))
    for _, label in dataset.samples:
        class_counts[label] += 1
    weights = 1.0 / class_counts
    sample_weights = torch.tensor([weights[label] for _, label in dataset.samples])
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def get_dataloaders(data_dir: str, val_split: float = 0.2, batch_size: int = 32, num_workers: int = 4):
    data_dir = Path(data_dir)

    full = datasets.ImageFolder(data_dir, transform=TRAIN_TRANSFORMS)
    n_val = int(len(full) * val_split)
    n_train = len(full) - n_val
    train_set, val_set = random_split(full, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))

    val_set.dataset = datasets.ImageFolder(data_dir, transform=VAL_TRANSFORMS)

    sampler = _make_weighted_sampler(full)
    # Only apply sampler to train indices
    train_sampler = WeightedRandomSampler(
        [sampler.weights[i] for i in train_set.indices],
        num_samples=n_train, replacement=True
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, full.classes
