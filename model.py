import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # output (B, 64, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # flatten from (B, 64, 1, 1) to (B, 64)
        return x


class RegressorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CNNBackbone()

        self.mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        # x shape: (B, 2, H, W) --> directly feed it to the backbone
        feat = self.backbone(x)  # (B, 32)
        output = self.mlp(feat)  # (B, 1)
        return output.squeeze(1)  # (B,)
