import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ----------------- Utils -----------------
def set_seed(seed: int):
    """Keeps runs reproducible for a given --seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_metas_check_channels(roots: list[Path]) -> dict:
    """Load meta.json from all roots and ensure 'channels' match for safe merge."""
    metas = []
    for r in roots:
        m = json.loads((r / "meta.json").read_text(encoding="utf-8"))
        metas.append(m)
    # Use the first as canonical; check equality of channels
    ch0 = metas[0].get("channels")
    if ch0 is None:
        raise RuntimeError(f"'channels' not found in {roots[0] / 'meta.json'}")
    for i, m in enumerate(metas[1:], start=1):
        chi = m.get("channels")
        if chi != ch0:
            raise RuntimeError(
                f"Channel mismatch between {roots[0]} and {roots[i]}.\n"
                f"{roots[0]}: {ch0}\n{roots[i]}: {chi}"
            )
    # Return the canonical meta (you can also merge other fields here if needed)
    return metas[0]


def list_all_cells(roots: list[Path]) -> list[Path]:
    """Scans each `root/cells/` directory and returns a list of all .npz files"""
    cells = []
    for r in roots:
        cdir = r / "cells"
        if not cdir.exists():
            raise FileNotFoundError(f"'cells' dir not found under {r}")
        cells.extend(sorted([p for p in cdir.iterdir() if p.suffix.lower() == ".npz"]))
    return cells


# ----------------- Dataset -----------------
class PerovCellTimepoints(Dataset):
    """
    Each timepoint of each cell is a sample.
    Augmentation is identical across all channels in a sample.
    """

    def __init__(
        self,
        cell_files: list[Path],
        channel_stats=None,
        predict=("soh_avg",),
        augment=True,
        soh_max: float | None = None,
        soh_min: float | None = None,
        drop_t0: bool = False,
    ):
        self.items = []
        self.cells = []
        self.predict = tuple(predict)
        self.augment = augment
        self.channel_stats = channel_stats
        self.soh_max = soh_max
        self.soh_min = soh_min

        for ci, cf in enumerate(cell_files):
            # loop over all cell_files (each is a .npz)
            dat = np.load(cf, allow_pickle=True)
            x = dat["x"].astype(np.float32)  # (T,C,H,W)
            T, C, H, W = x.shape

            # ----- targets dictionary -----
            targets = {}

            # SoH-related - PCE retention
            if "soh_avg" in dat.files:
                targets["soh_avg"] = dat["soh_avg"].astype(np.float32)
            # Retentions: voc, jsc, ff
            if "voc_ret" in dat.files:
                targets["voc_ret"] = dat["voc_ret"].astype(np.float32)
            if "jsc_ret" in dat.files:
                targets["jsc_ret"] = dat["jsc_ret"].astype(np.float32)
            if "ff_ret" in dat.files:
                targets["ff_ret"] = dat["ff_ret"].astype(np.float32)

            stack_code = int(dat["stack_code"]) if "stack_code" in dat.files else 0

            self.cells.append(
                {
                    "x": x,
                    "targets": targets,
                    "stack_code": stack_code,
                    "cell_file": str(cf),
                }
            )

            for ti in range(T):
                if drop_t0 and ti == 0:
                    continue
                # ci indexes the cell, ti indexes the timepoint inside that cell
                if (
                    self.soh_max is not None
                    and self.soh_min is not None
                    and "soh_avg" in targets
                ):
                    v = targets["soh_avg"][ti]
                    if np.isfinite(v) and v >= self.soh_max or v < self.soh_min:
                        continue
                self.items.append((ci, ti))

        self.mean = None
        self.std = None
        if channel_stats is not None:
            self.mean = torch.tensor(channel_stats["mean"], dtype=torch.float32).view(
                1, -1, 1, 1
            )
            self.std = torch.tensor(channel_stats["std"], dtype=torch.float32).view(
                1, -1, 1, 1
            )

    def __len__(self):
        """Returns the number of timepoints."""
        return len(self.items)

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        """
        Small geometric augmentation:
        - random translation (within 2 pixels, approximated in normalized coords
        - random rotation (within 2 degrees)
        """
        if not self.augment:
            return img
        C, H, W = img.shape
        max_shift = 2.0
        max_rot_deg = 2.0
        tx = (random.uniform(-max_shift, max_shift)) * 2.0 / (W - 1)
        ty = (random.uniform(-max_shift, max_shift)) * 2.0 / (H - 1)
        theta = math.radians(random.uniform(-max_rot_deg, max_rot_deg))
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        A = torch.tensor(
            [[cos_t, -sin_t, tx], [sin_t, cos_t, ty]], dtype=torch.float32
        ).unsqueeze(0)
        grid = F.affine_grid(A, size=(1, C, H, W), align_corners=False)
        img = F.grid_sample(
            img.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return img.squeeze(0)

    def __getitem__(self, idx):
        ci, ti = self.items[idx]
        cell = self.cells[ci]
        x = torch.from_numpy(cell["x"][ti])  # (C,H,W)

        # per-channel normalization
        if self.mean is not None and self.std is not None:
            x = (x - self.mean.squeeze(0)) / (self.std.squeeze(0) + 1e-6)
        # augmentation
        x = self._augment(x)

        y_dict = {}
        for k in self.predict:
            val = cell["targets"].get(k, None)
            y_dict[k] = (
                torch.tensor(val[ti], dtype=torch.float32)
                if val is not None
                else torch.tensor(float("nan"))
            )

        return {
            "x": x,
            "stack_code": torch.tensor(cell["stack_code"], dtype=torch.long),
            "y": y_dict,
            "cell_file": cell["cell_file"],
            "t_local": ti,
        }
