"""
CNN Baseline cho PathMNIST (28x28x3, 9 lớp).
Kiến trúc: 3 khối Conv-BN-ReLU-Pool → FC layers
"""

import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SimpleCNN(nn.Module):
    """
    Input : (B, 3, 28, 28)
    Block1: 3→32,  pool → (B, 32, 14, 14)
    Block2: 32→64, pool → (B, 64,  7,  7)
    Block3: 64→128, no pool → (B, 128, 7, 7)
    GAP   → (B, 128)
    FC    → 9
    """

    def __init__(self, num_classes: int = 9, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32,  pool=True),
            ConvBlock(32,  64,  pool=True),
            ConvBlock(64,  128, pool=False),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)
