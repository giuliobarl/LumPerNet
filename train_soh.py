"""
Train a SoH predictor from per-cell .npz datasets.

Expected dataset layout:
  dataset_root/
    meta.json
    cells/
      <cell_id>.npz  # contains x(T,C,H,W), soh_fw, [soh_rv], [soh_avg], stack_code

What this script does
- Splits by CELL (no leakage across timepoints), stratified by stack_code
- Computes per-channel mean/std on TRAIN ONLY
- Applies light, identical geometric augmentation across channels
- CNN + tiny stack embedding (SoHNet)
- Logs overall & per-stack MAE each epoch
- Saves: model.pt, channel_stats.npz, training_log.csv, metrics.json

Usage example:
  python train_soh.py \
    --data-root /path/to/dataset_root \
    --out-dir   ./runs/run1 \
    --predict soh_fw soh_rv \
    --epochs 60 --batch-size 128 --lr 3e-4 --val-split 0.2 \
    --seed 42
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ----------------- Utils -----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_meta(data_root: Path) -> dict:
    meta_path = data_root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found at {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_metas_check_channels(roots: list[Path]) -> dict:
    """Load meta.json from all roots and ensure 'channels' match."""
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
    ):
        self.items = []
        self.cells = []
        self.predict = tuple(predict)
        self.augment = augment
        self.channel_stats = channel_stats

        for ci, cf in enumerate(cell_files):
            dat = np.load(cf, allow_pickle=True)
            x = dat["x"].astype(np.float32)  # (T,C,H,W)
            T, C, H, W = x.shape
            targets = {"soh_avg": dat["soh_avg"].astype(np.float32)}
            if "soh_fw" in dat.files:
                targets["soh_fw"] = dat["soh_fw"].astype(np.float32)
            if "soh_rv" in dat.files:
                targets["soh_rv"] = dat["soh_rv"].astype(np.float32)
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
        return len(self.items)

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
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

        if self.mean is not None and self.std is not None:
            x = (x - self.mean.squeeze(0)) / (self.std.squeeze(0) + 1e-6)
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


# ----------------- Model -----------------
class TinyBackbone(nn.Module):
    def __init__(self, in_ch=9, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56->28
            nn.Conv2d(width, 2 * width, 3, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28->14
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


class TabularBranch(nn.Module):
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


class SoHNet(nn.Module):
    def __init__(self, n_stacks: int, in_ch=9, predict=("soh_fw",)):
        super().__init__()
        self.backbone = TinyBackbone(in_ch=in_ch, width=32)
        self.tab = TabularBranch(n_stacks=n_stacks, d_emb=4, n_cont=0)
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


# ----------------- Metrics -----------------
def masked_mae(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(true).float()
    return (torch.abs(pred - true) * mask).sum() / mask.sum().clamp_min(1.0)


def evaluate(model, loader, device, predict=("soh_fw",), per_stack=False):
    """
    Returns:
      overall: { "mae": {target: val}, "rmse": {...}, "r2": {...} }
      per_stack: same structure but per stack code if per_stack=True
    """
    import math

    model.eval()

    # accumulators per target
    mae_sum = {k: 0.0 for k in predict}
    se_sum = {k: 0.0 for k in predict}  # sum of squared errors
    y_sum = {k: 0.0 for k in predict}
    y2_sum = {k: 0.0 for k in predict}
    count = {k: 0 for k in predict}

    # per-stack accumulators: dict[target][stack] -> ...
    if per_stack:

        def ps():
            return {"mae": 0.0, "se": 0.0, "y": 0.0, "y2": 0.0, "n": 0}

        per_stack_acc = {k: {} for k in predict}
    else:
        per_stack_acc = None

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            sc = batch["stack_code"].to(device).long()
            out = model(x, sc)
            for k in predict:
                y = batch["y"][k].to(device)
                m = torch.isfinite(y)
                if m.sum() == 0:
                    continue
                y_m = y[m]
                pred = out[k][m]
                err = pred - y_m
                ae = torch.abs(err)
                se = err * err

                mae_sum[k] += float(ae.sum().item())
                se_sum[k] += float(se.sum().item())
                y_sum[k] += float(y_m.sum().item())
                y2_sum[k] += float((y_m * y_m).sum().item())
                count[k] += int(y_m.numel())

                if per_stack:
                    for s in torch.unique(sc[m]).tolist():
                        s = int(s)
                        sel = sc[m] == s
                        acc = per_stack_acc[k].setdefault(s, ps())
                        acc["mae"] += float(ae[sel].sum().item())
                        acc["se"] += float(se[sel].sum().item())
                        ym = y_m[sel]
                        acc["y"] += float(ym.sum().item())
                        acc["y2"] += float((ym * ym).sum().item())
                        acc["n"] += int(sel.sum().item())

    # overall metrics
    overall = {"mae": {}, "rmse": {}, "r2": {}}
    for k in predict:
        n = max(count[k], 1)
        mae = mae_sum[k] / n
        rmse = math.sqrt(se_sum[k] / n)
        denom = y2_sum[k] - (y_sum[k] ** 2) / n
        r2 = 1.0 - (se_sum[k] / denom) if denom > 0 else float("nan")
        overall["mae"][k] = mae
        overall["rmse"][k] = rmse
        overall["r2"][k] = r2

    # per-stack metrics if requested
    per_stack_metrics = {}
    if per_stack:
        per_stack_metrics = {
            metric: {k: {} for k in predict} for metric in ("mae", "rmse", "r2")
        }
        for k in predict:
            for s, acc in per_stack_acc[k].items():
                n = max(acc["n"], 1)
                mae = acc["mae"] / n
                rmse = math.sqrt(acc["se"] / n)
                denom = acc["y2"] - (acc["y"] ** 2) / n
                r2 = 1.0 - (acc["se"] / denom) if denom > 0 else float("nan")
                per_stack_metrics["mae"][k][s] = mae
                per_stack_metrics["rmse"][k][s] = rmse
                per_stack_metrics["r2"][k][s] = r2

    return overall, per_stack_metrics


# ----------------- Channel stats -----------------
def compute_channel_stats(cell_files: list[Path], sample_cap: int | None = None):
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

    print(f"Train cells: {train_cells}, Val cells: {val_cells}.")
    return train_cells, val_cells


# ----------------- Plotting -----------------
def parity_plot(model, loader, device, target: str, out_path: Path):
    """Save a parity (y_true vs y_pred) scatter for a given target."""
    model.eval()
    ys, yh = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            sc = batch["stack_code"].to(device).long()
            y = batch["y"][target].to(device)
            m = torch.isfinite(y)
            if m.any():
                pred = model(x, sc)[target][m]
                ys.append(y[m].cpu().numpy())
                yh.append(pred.cpu().numpy())
    if not ys:
        print(f"[WARN] No finite '{target}' values for parity plot.")
        return
    import numpy as np

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(yh)

    # Figure
    fig, ax = plt.subplots(figsize=(4, 4), dpi=160)
    ax.scatter(y_true, y_pred, s=6, alpha=0.6)
    lo, hi = float(np.nanmin([y_true.min(), y_pred.min()])), float(
        np.nanmax([y_true.max(), y_pred.max()])
    )
    pad = 0.02 * (hi - lo + 1e-8)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], lw=1.0)
    ax.set_xlabel(f"True {target}")
    ax.set_ylabel(f"Pred {target}")
    ax.set_title(f"Parity: {target}")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ----------------- Training -----------------
def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_roots = [Path(p) for p in args.data_roots]
    for r in data_roots:
        if not (r / "meta.json").exists():
            raise FileNotFoundError(f"meta.json not found in {r}")

    meta = load_metas_check_channels(data_roots)
    channels = meta["channels"]
    in_ch = len(channels)

    cells = list_all_cells(data_roots)
    if len(cells) == 0:
        raise RuntimeError("No .npz cells found in any provided data root.")

    # Split by cell, stratified by stack
    train_cells, val_cells = stratified_cell_split(cells, args.val_split, args.seed)

    # Channel stats on TRAIN ONLY
    ch_stats = compute_channel_stats(train_cells)
    np.savez(out_dir / "channel_stats.npz", **ch_stats)

    # Datasets
    ds_train = PerovCellTimepoints(
        train_cells,
        channel_stats=ch_stats,
        predict=tuple(args.predict),
        augment=not args.no_aug,
    )
    ds_val = PerovCellTimepoints(
        val_cells, channel_stats=ch_stats, predict=tuple(args.predict), augment=False
    )

    # Infer num stacks for embedding
    all_stacks = set()
    for cf in cells:
        dat = np.load(cf, allow_pickle=True)
        all_stacks.add(int(dat["stack_code"]) if "stack_code" in dat.files else 0)
    n_stacks = max(all_stacks) + 1 if len(all_stacks) > 0 else 1

    # Loaders
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Model, opt, sched
    model = SoHNet(n_stacks=n_stacks, in_ch=in_ch, predict=tuple(args.predict)).to(
        device
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    # Train
    import csv

    best_val = float("inf")
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        n_batches = 0
        for batch in dl_train:
            x = batch["x"].to(device)
            sc = batch["stack_code"].to(device).long()
            y = {k: batch["y"][k].to(device) for k in args.predict}

            out = model(x, sc)
            loss = 0.0
            for i, k in enumerate(args.predict):
                w = 1.0 if i == 0 else 0.5
                loss = loss + w * masked_mae(out[k], y[k])

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += float(loss.item())
            n_batches += 1

        sched.step()

        # Eval
        overall_tr, _ = evaluate(
            model, dl_train, device, predict=tuple(args.predict), per_stack=False
        )
        overall_va, per_stack_va = evaluate(
            model, dl_val, device, predict=tuple(args.predict), per_stack=True
        )

        row = {"epoch": epoch, "train_loss": tot_loss / max(n_batches, 1)}

        # Add MAE:
        for k, v in overall_tr["mae"].items():
            row[f"train_mae_{k}"] = v
        for k, v in overall_va["mae"].items():
            row[f"val_mae_{k}"] = v

        # Add RMSE:
        for k, v in overall_tr["rmse"].items():
            row[f"train_rmse_{k}"] = v
        for k, v in overall_va["rmse"].items():
            row[f"val_rmse_{k}"] = v

        # Add R^2:
        for k, v in overall_tr["r2"].items():
            row[f"train_r2_{k}"] = v
        for k, v in overall_va["r2"].items():
            row[f"val_r2_{k}"] = v

        log_rows.append(row)

        primary = f"val_mae_{args.predict[0]}"
        if row[primary] < best_val:
            best_val = row[primary]

            best_val_mae = overall_va["mae"]
            best_val_rmse = overall_va["rmse"]
            best_val_r2 = overall_va["r2"]
            best_tr_mae = overall_tr["mae"]
            best_tr_rmse = overall_tr["rmse"]
            best_tr_r2 = overall_tr["r2"]

            torch.save(
                {
                    "model": model.state_dict(),
                    "meta": meta,
                    "channel_stats": ch_stats,
                    "predict": list(args.predict),
                },
                out_dir / "model.pt",
            )

        va_str = " | ".join(
            [
                f"{k} MAE:{overall_va['mae'][k]:.4f} "
                f"RMSE:{overall_va['rmse'][k]:.4f} "
                f"R2:{overall_va['r2'][k]:.3f}"
                for k in args.predict
            ]
        )
        print(
            f"[{epoch:03d}] loss {row['train_loss']:.4f} | val {va_str} | best {best_val:.4f}"
        )

    # Save logs & metrics
    with open(out_dir / "training_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(log_rows[0].keys()))
        w.writeheader()
        w.writerows(log_rows)

    # Summarize last-epoch overall val metrics (as your "test" for now)
    final_metrics = {
        "predict": list(args.predict),
        "best_val_mae_primary": best_val,  # tracked on the first target
        "best_epoch": {
            "train": {
                "mae": best_tr_mae,
                "rmse": best_tr_rmse,
                "r2": best_tr_r2,
            },
            "val": {
                "mae": best_val_mae,
                "rmse": best_val_rmse,
                "r2": best_val_r2,
            },
        },
        "last_epoch": {
            "train": {
                "mae": overall_tr["mae"],
                "rmse": overall_tr["rmse"],
                "r2": overall_tr["r2"],
            },
            "val": {
                "mae": overall_va["mae"],
                "rmse": overall_va["rmse"],
                "r2": overall_va["r2"],
            },
        },
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(final_metrics, indent=2), encoding="utf-8"
    )

    print(f"Done. Best model saved to {out_dir/'model.pt'}")

    # Reload best and make parity plot(s) on validation split
    ckpt = torch.load(out_dir / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    primary = args.predict[0]
    parity_plot(
        model,
        dl_val,
        device,
        target=primary,
        out_path=out_dir / f"parity_val_{primary}.png",
    )
    print(f"Saved parity plot to {out_dir / f'parity_val_{primary}.png'}")


# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-roots",
        nargs="+",
        default=None,
        help="One or more dataset roots, each with meta.json and cells/",
    )
    ap.add_argument(
        "--data-parent",
        type=str,
        default=None,
        help="Parent folder containing multiple dataset roots (each with meta.json and cells/).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and logs",
    )
    ap.add_argument(
        "--predict",
        nargs="+",
        default=["soh_fw"],
        choices=["soh_fw", "soh_rv", "soh_avg"],
        help="Targets to predict (order sets loss weights: first=1.0, others=0.5).",
    )
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--cpu", action="store_true", help="Force CPU even if CUDA is available"
    )
    ap.add_argument(
        "--no-aug", action="store_true", help="Disable geometric augmentation"
    )
    args = ap.parse_args()

    # Build data_roots from --data-roots plus optional --data-parent
    data_roots = []
    if args.data_roots:
        data_roots.extend([Path(p) for p in args.data_roots])
    if args.data_parent:
        parent = Path(args.data_parent)
        subs = [d for d in parent.iterdir() if d.is_dir()]
        # keep only valid dataset roots (must have meta.json and cells/)
        subs = [
            d for d in subs if (d / "meta.json").exists() and (d / "cells").exists()
        ]
        data_roots.extend(subs)

    # de-duplicate and sanity check
    args.data_roots = sorted(set(map(Path, data_roots)))
    if not args.data_roots:
        raise RuntimeError(
            "No dataset roots found. Provide --data-roots and/or --data-parent."
        )

    train(args)


if __name__ == "__main__":
    main()
