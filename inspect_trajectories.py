"""
Inspect true vs predicted SoH trajectories for representative cells.

This script:
- loads a trained SoH regressor
- runs inference on a dataset (typically validation)
- plots true vs predicted SoH vs time
  for selected representative cells
"""

import argparse
from pathlib import Path
import json
import numpy as np
import random
import torch
from torch.utils.data import DataLoader

from dataset import (
    load_metas_check_channels,
    list_all_cells,
    PerovCellTimepoints,
)
from model import SoHNet
from utils_data import stratified_cell_split
from utils_plot import plot_representative_cells

# ----------------- Utils -----------------
def set_seed(seed: int):
    """Keeps runs reproducible for a given --seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def select_representative_cells(cell_files, y_dict):
    """
    Selects three cells based on mean true SoH:
    - healthy: highest mean
    - borderline: closest to 1.0
    - degraded: lowest mean
    Works whether cell_files are Path or str.
    """
    stats = []

    for cf in cell_files:
        cf_key = str(cf)  # normalize Path -> str
        y = y_dict.get(cf_key)
        if y is None or len(y) == 0:
            continue
        stats.append((cf_key, float(np.mean(y))))

    if len(stats) < 3:
        raise RuntimeError(
            f"Not enough valid cells to select representatives. "
            f"Found {len(stats)} cells with finite values."
        )

    stats.sort(key=lambda x: x[1])  # ascending mean SoH
    degraded = stats[0][0]
    healthy = stats[-1][0]
    borderline = min(stats, key=lambda x: abs(x[1] - 1.0))[0]

    return healthy, borderline, degraded

def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- dataset roots ---
    data_roots = []
    parent = Path(args.data_parent)
    subs = [d for d in parent.iterdir() if d.is_dir()]
    # keep only valid dataset roots (must have meta.json and cells/)
    subs = [
        d for d in subs if (d / "meta.json").exists() and (d / "cells").exists()
    ]
    data_roots.extend(subs)
    meta = load_metas_check_channels(data_roots)
    channels = meta["channels"]

    # --- list all cells ---
    all_cells = list_all_cells(data_roots)

    # --- split by cell (same logic as training) ---
    train_cells, val_cells = stratified_cell_split(
        all_cells,
        val_split=args.val_split,
        seed=args.seed,
    )

    if len(val_cells) == 0:
        raise RuntimeError("Validation set is empty.")

    # --- load checkpoint ---
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    predict = ckpt["predict"]
    ch_stats = ckpt["channel_stats"]

    # --- validation dataset only (NO augmentation) ---
    ds_val = PerovCellTimepoints(
        val_cells,
        channel_stats=ch_stats,
        predict=tuple(predict),
        augment=False,
        soh_max=1.2,
        soh_min=0.8
    )

    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # --- infer number of stacks ---
    all_stacks = set()
    for cf in val_cells:
        dat = np.load(cf, allow_pickle=True)
        all_stacks.add(int(dat["stack_code"]) if "stack_code" in dat.files else 0)
    n_stacks = max(all_stacks) + 1

    # --- build and load model ---
    model = SoHNet(
        n_stacks=n_stacks,
        in_ch=len(channels),
        predict=tuple(predict),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    # --- collect true SoH per VALIDATION cell ---
    true_soh = {}
    for batch in dl_val:
        cell_files = batch["cell_file"]
        y = batch["y"]["soh_avg"]
        for i, cf in enumerate(cell_files):
            v = y[i].item()
            if not np.isfinite(v):
                continue
            true_soh.setdefault(cf, []).append(v)

    # --- select representative VALIDATION cells ---
    healthy, borderline, degraded = select_representative_cells(
        val_cells, true_soh
    )

    print("Selected validation cells:")
    print("  healthy   :", healthy)
    print("  borderline:", borderline)
    print("  degraded  :", degraded)

    # --- plot trajectories ---
    plot_representative_cells(
        model,
        dl_val,
        device,
        cell_files=[healthy, borderline, degraded],
        out_dir=out_dir,
        target="soh_avg",
        y_range=(0.8, 1.2),
        prefix="trajectory_val",
    )

    print(f"Trajectory plots saved to {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data-parent",
        type=str,
        required=True,
        help="Dataset parent (same as training)",
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (model.pt)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for trajectory plots",
    )

    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")

    args = ap.parse_args()
    
    main(args)
