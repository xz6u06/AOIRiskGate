from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, n_class: int):
        super().__init__()
        self.con1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.con2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.con3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.con1_dropout = nn.Dropout2d(p=0.3)
        self.con2_dropout = nn.Dropout2d(p=0.3)
        self.con3_dropout = nn.Dropout2d(p=0.3)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.con1(x)))
        x = self.con1_dropout(x)
        x = self.pool2(F.relu(self.con2(x)))
        x = self.con2_dropout(x)
        x = self.pool3(F.relu(self.con3(x)))
        x = self.con3_dropout(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def _build_resnet18(n_class: int, *, pretrained: bool = False) -> nn.Module:
    """ResNet-18 backbone.

    If pretrained=True, loads ImageNet weights (requires torchvision).
    """

    from torchvision import models

    if pretrained:

        try:
            weights = models.ResNet18_Weights.DEFAULT
            m = models.resnet18(weights=weights)
        except Exception:
            m = models.resnet18(pretrained=True)
    else:
        try:
            m = models.resnet18(weights=None)
        except Exception:
            m = models.resnet18(pretrained=False)
    """ 新版 torchvision（大概 0.13+ 之後）用 weights=... 的寫法
        舊版 torchvision 用 pretrained=True 這種參數
        所以寫了 try/except 兼容"""

    # Replace classification head
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, n_class)
    return m


def build_model(name: str, n_class: int, *, pretrained: bool = False) -> nn.Module:
    name = name.lower()
    if name in {"simple", "simplecnn", "cnn"}:
        return SimpleCNN(n_class)
    if name in {"resnet18", "r18"}:
        return _build_resnet18(n_class, pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}")
