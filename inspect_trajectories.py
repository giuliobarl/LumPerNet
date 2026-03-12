"""
Inspect true vs predicted SoH trajectories for representative cells.

This script:
- loads a trained SoH regressor
- runs inference on a dataset (typically validation)
- plots true vs predicted SoH vs time
  for selected representative cells
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PerovCellTimepoints, list_all_cells, load_metas_check_channels
from models import LumPerNet
from utils_data import stratified_cell_split
from utils_plot import plot_absolute_error_vs_time, plot_multiple_ensemble_trajectories


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


def collect_ensemble_timeseries(models, test_cells, args, device):

    ensemble = defaultdict(lambda: defaultdict(list))
    true_values = defaultdict(dict)
    absolute_time = defaultdict(dict)

    for model, ch_stats in models:

        ds_test = PerovCellTimepoints(
            test_cells,
            channel_stats=ch_stats,
            predict=("soh_avg",),
            augment=False,
            soh_max=args.soh_max,
            soh_min=args.soh_min,
        )

        dl_test = DataLoader(
            ds_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

        model.eval()

        with torch.no_grad():
            for batch in dl_test:
                x = batch["x"].to(device)
                sc = batch["stack_code"].to(device)
                t_local = batch["t_local"].cpu().numpy()
                t_idx = batch["t_idx"].cpu().numpy()
                cell_files = batch["cell_file"]
                y_true = batch["y"]["soh_avg"].cpu().numpy()
                y_pred = model(x, sc)["soh_avg"].cpu().numpy()

                for i, cf in enumerate(cell_files):
                    local_t = t_local[i]
                    abs_t = t_idx[i]
                    if not np.isfinite(y_true[i]):
                        continue

                    ensemble[cf][local_t].append(y_pred[i])
                    true_values[cf][local_t] = y_true[i]
                    absolute_time[cf][local_t] = abs_t

    # build final structure
    out = {}

    for cf in ensemble:
        local_times = sorted(ensemble[cf].keys())
        # use stored absolute acquisition indices
        times_idx = np.array([absolute_time[cf][t] for t in local_times])

        # convert to hours
        times_abs = np.array(times_idx) * 4.0 / 60.0

        y_true = np.array([true_values[cf][t] for t in local_times])

        y_mean = np.array([np.mean(ensemble[cf][t]) for t in local_times])

        y_std = np.array([np.std(ensemble[cf][t]) for t in local_times])

        out[cf] = {
            "t": np.array(times_abs),
            "y_true": y_true,
            "y_pred_mean": y_mean,
            "y_pred_std": y_std,
        }

    return out


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
    subs = [d for d in subs if (d / "meta.json").exists() and (d / "cells").exists()]
    data_roots.extend(subs)
    meta = load_metas_check_channels(data_roots)
    channels = meta["channels"]

    # --- list all cells ---
    all_cells = list_all_cells(data_roots)

    # --- split by cell (same logic as training) ---
    # Always inspect fixed TEST set
    _, test_cells = stratified_cell_split(all_cells, args.test_split, args.seed)
    # train_cells = [c for i, f in enumerate(cv_folds) if i != args.fold_id for c in f]

    ckpt_dir = Path(args.checkpoints_dir)
    ckpt_paths = sorted(ckpt_dir.glob("fold_*/model_*.pt"))

    if len(ckpt_paths) == 0:
        raise RuntimeError("No model_*.pt checkpoints found.")

    models = []
    predict = None

    for p in ckpt_paths:
        ckpt = torch.load(p, map_location=device, weights_only=False)

        if predict is None:
            predict = ckpt["predict"]

        ch_stats = ckpt["channel_stats"]

        # infer number of stacks
        all_stacks = set()
        for cf in test_cells:
            dat = np.load(cf, allow_pickle=True)
            all_stacks.add(int(dat["stack_code"]) if "stack_code" in dat.files else 0)
        n_stacks = max(all_stacks) + 1

        model = LumPerNet(
            n_stacks=n_stacks,
            in_ch=len(channels),
            predict=tuple(predict),
            use_stack=False,
        ).to(device)

        model.load_state_dict(ckpt["model"])
        model.eval()

        models.append((model, ch_stats))

    cell_data = collect_ensemble_timeseries(models, test_cells, args, device)

    # -------------------------------
    # Absolute error distribution vs time (pooled across all test samples)
    # -------------------------------
    t_all = []
    abs_all = []

    for cf, d in cell_data.items():
        t = d["t"]  # already elapsed hours
        y_true = d["y_true"]
        y_pred = d["y_pred_mean"]

        m = np.isfinite(t) & np.isfinite(y_true) & np.isfinite(y_pred)
        if m.sum() == 0:
            continue

        t = t[m]
        y_true = y_true[m]
        y_pred = y_pred[m]

        abs_err = np.abs(y_pred - y_true)

        t_all.append(t)
        abs_all.append(abs_err)

    if len(t_all) > 0:
        t_all = np.concatenate(t_all)
        abs_all = np.concatenate(abs_all)

        out_path_abs = out_dir / "abs_error_vs_time_test.pdf"
        plot_absolute_error_vs_time(
            t_all,
            abs_all,
            out_path_abs,
            gridsize=30,
        )
        print(f"Saved relative error vs time plot to {out_path_abs}")
    else:
        print("[WARN] No data to plot relative error vs time.")

    # print(f"Plotting trajectories for {len(cell_data)} test cells...")

    # for cf in cell_data.keys():
    #     stem = Path(cf).stem
    #     out_path = out_dir / f"trajectory_test_{stem}.png"
    #     plot_ensemble_trajectory(cell_data, cf, out_path)

    selected_cells = [
        r"data\final\2026-01-30\cells\batch_1_4D.npz",
        r"data\final\2026-01-30\cells\batch_1_6C.npz",
        r"data\final\2026-01-28\cells\batch_0_3B.npz",
    ]

    # Important: ensure keys match exactly what is stored
    selected_cells = [str(Path(c)) for c in selected_cells]

    out_path = Path(args.out_dir) / "trajectories_selected_cells_hor.png"

    plot_multiple_ensemble_trajectories(
        cell_data,
        selected_cells,
        out_path,
        y_range=(args.soh_min - 0.05, args.soh_max + 0.05),
    )

    print(f"Saved combined trajectory figure to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data-parent",
        type=str,
        required=True,
        help="Dataset parent (same as training)",
    )
    ap.add_argument(
        "--checkpoints-dir",
        type=str,
        required=True,
        help="Directory containing fold checkpoints (model_0.pt, model_1.pt, ...)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for trajectory plots",
    )
    ap.add_argument("--soh-max", type=float, default=2.0)
    ap.add_argument("--soh-min", type=float, default=0.0)
    ap.add_argument("--n-folds", type=int, required=True)
    ap.add_argument("--test-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cpu", action="store_true")

    args = ap.parse_args()

    main(args)
