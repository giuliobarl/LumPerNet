"""
Train a SoH predictor from per-cell .npz datasets.

Expected dataset layout:
  dataset_root/
    meta.json
    cells/
      <cell_id>.npz  

Each .npz file corresponds to one cell, containing:
- x(T,C,H,W): T = timestamp, C = channels, (H, W) = spatial size
- stack_code: integer cell ID
- targets: soh_avg = pce_ret, voc_ret, jsc_ret, ff_ret

What this script does:
- Loads all per-cell files
- Splits by CELL (no leakage across timepoints), stratified by stack_code
- Computes per-channel mean/std on TRAIN ONLY
- Applies light, identical geometric augmentation across channels
- Builds a CNN-based multi-task regressor (SoHNet)
- Trains it, with optional consistency loss between SoH and Voc/Jsc/FF retentions
- Logs overall MAE/RMSE/R2 each epoch
- Optionally:
    * soft consistency loss between soh_avg and product of voc/jsc/ff retentions
- Saves: model.pt, channel_stats.npz, training_log.csv, metrics.json
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PerovCellTimepoints, load_metas_check_channels, list_all_cells
from model import SoHNet
from utils_data import *
from utils_plot import *


# ----------------- Utils -----------------
def set_seed(seed: int):
    """Keeps runs reproducible for a given --seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------- Metrics -----------------
def masked_mae(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Computes MAE while ignoring NaNs."""
    mask = torch.isfinite(true).float()
    return (torch.abs(pred - true) * mask).sum() / mask.sum().clamp_min(1.0)


def evaluate_regression(model, loader, device, predict=("soh_avg",), per_stack=False):
    """
    Returns:
      overall: { "mae": {target: val}, "rmse": {...}, "r2": {...} }
      per_stack: same structure but per stack code if per_stack=True
    """
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


def compute_batch_loss(
    outputs,
    targets,
    predict,
    consistency_weight=0.0,
    consistency_log=False,
):
    """
    Computes the scalar loss for one batch.
    """
    loss = 0.0

    # base multi-task MAE
    for i, k in enumerate(predict):
        w = 1.0 if i == 0 else 0.5
        loss = loss + w * masked_mae(outputs[k], targets[k])

    # optional consistency loss
    if consistency_weight > 0.0:
        s_hat = outputs["soh_avg"]
        rv_hat = outputs["voc_ret"]
        rj_hat = outputs["jsc_ret"]
        rf_hat = outputs["ff_ret"]

        eps = 1e-6
        if consistency_log:
            cons = torch.nn.functional.l1_loss(
                torch.log(s_hat.clamp_min(eps)),
                torch.log(rv_hat.clamp_min(eps))
                + torch.log(rj_hat.clamp_min(eps))
                + torch.log(rf_hat.clamp_min(eps)),
            )
        else:
            cons = torch.nn.functional.l1_loss(
                s_hat, rv_hat * rj_hat * rf_hat
            )

        loss = loss + consistency_weight * cons

    return loss


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
        soh_max=args.soh_max,
        soh_min=args.soh_min,
        drop_t0=True
    )
    ds_val = PerovCellTimepoints(
        val_cells,
        channel_stats=ch_stats,
        predict=tuple(args.predict),
        augment=False,
        soh_max=args.soh_max,
        soh_min=args.soh_min,
        drop_t0=True
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

    # Target stats
    train_y = collect_true_targets(dl_train, args.predict)
    val_y = collect_true_targets(dl_val, args.predict)

    target_stats = {
        "train": summarize_targets(train_y),
        "val": summarize_targets(val_y),
    }

    (out_dir / "target_stats.json").write_text(
        json.dumps(target_stats, indent=2),
        encoding="utf-8",
)

    plot_target_histograms(
        train_y,
        out_dir=out_dir,
        split="train",
    )

    plot_target_histograms(
        val_y,
        out_dir=out_dir,
        split="val",
    )

    # Model, opt, sched
    model = SoHNet(n_stacks=n_stacks, in_ch=in_ch, predict=tuple(args.predict)).to(
        device
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    # Consistency setup
    use_consistency = (
        args.consistency_weight > 0.0
        and "soh_avg" in args.predict
        and "voc_ret" in args.predict
        and "jsc_ret" in args.predict
        and "ff_ret" in args.predict
    )
    if use_consistency:
        print(
            f"Using consistency loss (weight={args.consistency_weight}, "
            f"log-space={args.consistency_log}) on soh_avg vs voc_ret*jsc_ret*ff_ret"
        )

    # Train
    best_val = float("inf")
    best_epoch = None
    best_metrics_tr = None
    best_metrics_va = None
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for batch in dl_train:
            x = batch["x"].to(device)
            sc = batch["stack_code"].to(device).long()
            y = {k: batch["y"][k].to(device) for k in args.predict}
            out = model(x, sc)

            # Supervised multi-task loss
            loss = compute_batch_loss(
                out,
                y,
                predict=args.predict,
                consistency_weight=args.consistency_weight,
                consistency_log=args.consistency_log,
            )

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += float(loss.item())
            n_batches += 1

        train_loss_epoch = train_loss_sum / max(n_batches, 1)

        sched.step()

        # Eval
        model.eval()
        val_loss_sum = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dl_val:
                x = batch["x"].to(device)
                sc = batch["stack_code"].to(device).long()
                y = {k: batch["y"][k].to(device) for k in args.predict}

                outputs = model(x, sc)

                loss = compute_batch_loss(
                    outputs,
                    y,
                    predict=args.predict,
                    consistency_weight=args.consistency_weight,
                    consistency_log=args.consistency_log,
                )

                val_loss_sum += loss.item()
                n_batches += 1

        val_loss_epoch = val_loss_sum / max(n_batches, 1)

        overall_tr, _ = evaluate_regression(
            model, dl_train, device, predict=tuple(args.predict), per_stack=False
        )
        overall_va, per_stack_va = evaluate_regression(
            model, dl_val, device, predict=tuple(args.predict), per_stack=True
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss_epoch,
            "val_loss": val_loss_epoch
            }

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
        metric_for_best = row[primary]
        va_str = " ".join([f"{k}:{overall_va['mae'][k]:.4f}" for k in args.predict])
        print(
            f"[{epoch:03d}] loss {row['train_loss']:.4f} | val {va_str} | best {best_val:.4f}"
        )

        if metric_for_best < best_val:
            best_val = metric_for_best
            best_epoch = epoch
            best_metrics_tr = overall_tr
            best_metrics_va = overall_va

            torch.save(
                {
                    "model": model.state_dict(),
                    "meta": meta,
                    "channel_stats": ch_stats,
                    "predict": list(args.predict),
                },
                out_dir / "model.pt",
            )

    # Save logs & metrics
    if log_rows:
        import csv

        with open(out_dir / "training_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)

        plot_loss_history(
            log_rows,
            out_dir / "loss_history.png",
        )
    else:
        print("[WARN] No log rows recorded; skipping training_log.csv and loss_history.png")

    final_metrics = {
        "predict": list(args.predict),
        "best_epoch": best_epoch,
        "best_val_mae_primary": best_val,
        "best_train": best_metrics_tr,  # contains mae/rmse/r2 per target
        "best_val": best_metrics_va,
        "consistency_weight": args.consistency_weight,
        "consistency_log": args.consistency_log,
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(final_metrics, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Best model saved to {out_dir/'model.pt'}")

    ckpt = torch.load(out_dir / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    parity_plots_all(
        model,
        dl_val,
        device,
        targets=args.predict,
        out_dir=out_dir,
    )
    print(f"Saved parity plots to {out_dir}")


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
        default=["soh_avg"],
        choices=[
            "soh_avg",
            "soh_fw",
            "soh_rv",
            "voc_ret",
            "jsc_ret",
            "ff_ret",
        ],
        help=(
            "Targets to predict. Order sets loss weights: first=1.0, "
            "others=0.5. For consistency loss, you want: "
            "soh_avg voc_ret jsc_ret ff_ret"
        ),
    )
    ap.add_argument(
        "--soh-max",
        type=float,
        default=None,
        help="If set, discard samples with soh_avg >= this value (e.g. 1.2).",
    )
    ap.add_argument(
        "--soh-min",
        type=float,
        default=None,
        help="If set, discard samples with soh_avg < this value (e.g. 0.8).",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--cpu", action="store_true", help="Force CPU even if CUDA is available"
    )
    ap.add_argument(
        "--no-aug", action="store_true", help="Disable geometric augmentation"
    )

    # NEW: consistency-loss options
    ap.add_argument(
        "--consistency-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for consistency loss between soh_avg and "
            "voc_ret*jsc_ret*ff_ret (0 disables)."
        ),
    )
    ap.add_argument(
        "--consistency-log",
        action="store_true",
        help="If set, enforce consistency in log-space: log SoH ≈ sum(log retentions).",
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

"""
example bash usage

python train_regressor.py \
  --data-parent ./data/final \
  --out-dir ./runs/soh_multitask \
  --predict soh_avg voc_ret jsc_ret ff_ret \
  --soh-min 0.8 \
  --soh-max 1.2 \
  --epochs 100 \
  --batch-size 64 \
  --lr 3e-4 \
  --val-split 0.2 \
  --seed 42 \
  --consistency-weight 0.5 \
  --consistency-log
"""