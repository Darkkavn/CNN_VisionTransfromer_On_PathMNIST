"""
ResNet18 transfer learning cho PathMNIST.
Dùng pretrained weights từ ImageNet, thay FC layer cuối → 9 lớp.
"""

import torch.nn as nn
from torchvision import models


class ResNet18PathMNIST(nn.Module):
    """
    ResNet18 pretrained → fine-tune toàn bộ.
    FC layer: 512 → num_classes.
    """

    def __init__(self, num_classes: int = 9, pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.model = backbone

    def forward(self, x):
        return self.model(x)
