"""
EfficientNet-B0 classifier — same architecture as the coastal species project,
scaled to num_classes for the combined F4K + Mediterranean dataset.
"""

import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, feature_extract: bool = False) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
