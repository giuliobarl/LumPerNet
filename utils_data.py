import random
from pathlib import Path

import numpy as np
import torch


# ----------------- Utils -----------------
def set_seed(seed: int):
    """Keeps runs reproducible for a given --seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------- Channel stats -----------------
def compute_channel_stats(cell_files: list[Path], sample_cap: int | None = None):
    """Estimate global per-channel mean and std on the training cells only."""
    sum_c = None
    sumsq_c = None
    n = 0
    cnt = 0
    for cf in cell_files:
        dat = np.load(cf, allow_pickle=True)
        X = dat["x"].astype(np.float32)  # (T,C,H,W)
        T, C, H, W = X.shape
        if sum_c is None:
            sum_c = np.zeros(C, dtype=np.float64)
            sumsq_c = np.zeros(C, dtype=np.float64)
        for t in range(T):
            x = X[t]
            sum_c += x.reshape(C, -1).mean(axis=1)
            sumsq_c += (x.reshape(C, -1) ** 2).mean(axis=1)
            n += 1
            cnt += 1
            if sample_cap is not None and cnt >= sample_cap:
                break
        if sample_cap is not None and cnt >= sample_cap:
            break
    mean = (sum_c / max(n, 1)).astype(np.float32)
    var = (sumsq_c / max(n, 1)) - (mean.astype(np.float64) ** 2)
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    return {"mean": mean, "std": std}


# ----------------- Split logic -----------------
def stratified_cell_split(cell_files: list[Path], val_split: float, seed: int):
    """Splits by cell (never splits a cell's timepoints between train and val) and stratifies by stack_code."""
    rng = random.Random(seed)
    by_stack = {}
    for cf in cell_files:
        dat = np.load(cf, allow_pickle=True)
        s = int(dat["stack_code"]) if "stack_code" in dat.files else 0
        by_stack.setdefault(s, []).append(cf)
    train_cells, val_cells = [], []
    for s, files in by_stack.items():
        files = files[:]
        rng.shuffle(files)
        n_val = (
            max(1, int(round(len(files) * val_split)))
            if len(files) > 1
            else (1 if val_split > 0 else 0)
        )
        val_cells.extend(files[:n_val])
        train_cells.extend(files[n_val:])
    if len(train_cells) == 0 and len(val_cells) > 0:
        train_cells.append(val_cells.pop())

    # print(f"Train cells: {train_cells}, Val cells: {val_cells}.")
    return train_cells, val_cells

# ----------------- Target stats -----------------
def collect_true_targets(loader, targets):
    """
    Collects ground-truth y values only (no predictions).
    Returns: dict[target] -> np.ndarray
    """
    data = {k: [] for k in targets}

    for batch in loader:
        for k in targets:
            y = batch["y"][k]
            m = torch.isfinite(y)
            if m.any():
                data[k].append(y[m].cpu().numpy())

    for k in targets:
        data[k] = np.concatenate(data[k]) if data[k] else None

    return data

def summarize_targets(data):
    stats = {}
    for k, y in data.items():
        if y is None:
            continue
        stats[k] = {
            "n": int(y.size),
            "mean": float(np.mean(y)),
            "std": float(np.std(y)),
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "p01": float(np.percentile(y, 1)),
            "p05": float(np.percentile(y, 5)),
            "p50": float(np.percentile(y, 50)),
            "p95": float(np.percentile(y, 95)),
            "p99": float(np.percentile(y, 99)),
        }
    return stats