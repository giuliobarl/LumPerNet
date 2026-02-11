import random

import numpy as np
import torch
import torch.nn as nn


# ----------------- Utils -----------------
def set_seed(seed: int):
    """Keeps runs reproducible for a given --seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------- Tiny Model -----------------
class TinyBackbone(nn.Module):
    """Simple CNN branch: shared spatial feature extractor for all tasks."""

    def __init__(self, in_ch=9, width=16, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = 2 * width

    def forward(self, x):
        return self.net(x).flatten(1)


class TabularBranch(nn.Module):
    def __init__(self, n_stacks: int, d_emb: int = 2, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(max(n_stacks, 1), d_emb)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = d_emb

    def forward(self, stack_code, cont_feats=None):
        return self.dropout(self.emb(stack_code))


class SoHNet(nn.Module):
    """Main multi-task model, composed of TinyBackbone and TabularBranch."""

    def __init__(
        self, n_stacks: int, in_ch=9, predict=("soh_avg",), use_stack: bool = True
    ):
        super().__init__()
        self.backbone = TinyBackbone(in_ch=in_ch, width=16)

        self.use_stack = use_stack
        if use_stack:
            self.tab = TabularBranch(n_stacks=n_stacks, d_emb=2)
            fusion_dim = self.backbone.out_dim + self.tab.out_dim
        else:
            self.tab = None
            fusion_dim = self.backbone.out_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, len(predict)),
        )
        self.predict = predict

    def forward(self, imgs, stack_code, cont_feats=None):
        fi = self.backbone(imgs)

        if self.use_stack:
            ft = self.tab(stack_code, cont_feats)
            x = torch.cat([fi, ft], dim=1)
        else:
            x = fi

        out = self.head(x)
        return {k: out[:, i] for i, k in enumerate(self.predict)}


# ----------------- Large Model -----------------
class LargeBackbone(nn.Module):
    """Simple CNN branch: shared spatial feature extractor for all tasks."""

    def __init__(self, in_ch=9, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * width, 4 * width, 3, padding=1),
            nn.BatchNorm2d(4 * width),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * width, 4 * width, 3, padding=1),
            nn.BatchNorm2d(4 * width),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = 4 * width

    def forward(self, x):
        return self.net(x).flatten(1)


class LargeTabularBranch(nn.Module):
    """Simple fully-connected branch handling non-image features (e.g., stack_code)."""

    def __init__(self, n_stacks: int, d_emb: int = 4, n_cont: int = 0):
        super().__init__()
        self.emb = nn.Embedding(max(n_stacks, 1), d_emb)
        self.mlp = nn.Sequential(
            nn.Linear(d_emb + n_cont, 32),
            nn.ReLU(inplace=True),
        )
        self.out_dim = 32

    def forward(self, stack_code: torch.Tensor, cont_feats: torch.Tensor | None = None):
        e = self.emb(stack_code)
        c = (
            torch.zeros(e.size(0), 0, device=e.device)
            if cont_feats is None
            else cont_feats
        )
        return self.mlp(torch.cat([e, c], dim=1))


class LargeSoHNet(nn.Module):
    """Main multi-task model, composed of LargeBackbone and LargeTabularBranch."""

    def __init__(self, n_stacks: int, in_ch=9, predict=("soh_avg",)):
        super().__init__()
        self.backbone = LargeBackbone(in_ch=in_ch, width=32)
        self.tab = LargeTabularBranch(n_stacks=n_stacks, d_emb=4, n_cont=0)
        fusion_dim = self.backbone.out_dim + self.tab.out_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, len(predict)),
        )
        self.predict = predict

    def forward(self, imgs, stack_code, cont_feats=None):
        fi = self.backbone(imgs)
        ft = self.tab(stack_code, cont_feats)
        out = self.head(torch.cat([fi, ft], dim=1))
        return {k: out[:, i] for i, k in enumerate(self.predict)}


# ----------------- Baseline Model -----------------
class BaselineMLP(nn.Module):
    """
    Baseline regressor:
      - space-averages the C channels (no spatial info)
      - concatenates with same stack embedding branch used in SoHNet
      - shallow MLP head
    """

    def __init__(
        self,
        n_stacks: int,
        in_ch: int,
        predict=("soh_avg",),
        use_stack: bool = True,
        d_emb: int = 2,
        hidden=(64, 32),
        dropout: float = 0.0,
    ):
        super().__init__()

        self.use_stack = use_stack
        if use_stack:
            # same stack encoding logic as SoHNet
            self.tab = TabularBranch(n_stacks=n_stacks, d_emb=d_emb)
            in_dim = in_ch + self.tab.out_dim
        else:
            self.tab = None
            in_dim = in_ch

        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, len(predict)))

        self.head = nn.Sequential(*layers)
        self.predict = tuple(predict)

    def forward(self, imgs, stack_code, cont_feats=None):
        # imgs: [B, C, H, W]
        x_mean = imgs.mean(dim=(2, 3))  # [B, C]
        if self.use_stack:
            ft = self.tab(stack_code, cont_feats)  # [B, tab_dim]
            x = torch.cat([x_mean, ft], dim=1)
        else:
            x = x_mean

        out = self.head(x)
        return {k: out[:, i] for i, k in enumerate(self.predict)}
