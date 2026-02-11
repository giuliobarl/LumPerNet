import argparse
import copy
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import PerovCellTimepoints, list_all_cells, load_metas_check_channels
from models import BaselineMLP
from utils_data import (
    collect_true_targets,
    compute_channel_stats,
    stratified_cell_split,
    stratified_kfold_cells,
)
from utils_plot import (
    parity_plots_all,
    plot_loss_history,
    plot_target_histograms,
    visualise_dataloader_soh,
)


# ----------------- Utils -----------------
def set_seed(seed: int):
    """Keeps runs reproducible for a given --seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------- Dataset -----------------
def build_datasets_and_loaders(args, fold_id: int):
    data_roots = [Path(p) for p in args.data_roots]
    for r in data_roots:
        if not (r / "meta.json").exists():
            raise FileNotFoundError(f"meta.json not found in {r}")

    meta = load_metas_check_channels(data_roots)
    channels = meta["channels"]

    cells = list_all_cells(data_roots)
    if len(cells) == 0:
        raise RuntimeError("No .npz cells found.")

    # Fixed test split
    cv_cells, test_cells = stratified_cell_split(cells, args.test_split, args.seed)

    # CV folds on remaining cells
    cv_folds = stratified_kfold_cells(cv_cells, args.n_folds, args.seed)

    if fold_id < 0 or fold_id >= args.n_folds:
        raise ValueError(f"Invalid fold_id {fold_id}")

    val_cells = cv_folds[fold_id]
    train_cells = [c for i, f in enumerate(cv_folds) if i != fold_id for c in f]

    assert set(train_cells).isdisjoint(val_cells)
    assert set(train_cells).isdisjoint(test_cells)
    assert set(val_cells).isdisjoint(test_cells)

    # Channel stats on TRAIN only
    ch_stats = compute_channel_stats(train_cells)

    # Datasets
    ds_train = PerovCellTimepoints(
        train_cells,
        channel_stats=ch_stats,
        predict=tuple(args.predict),
        augment=not args.no_aug,
        soh_max=args.soh_max,
        soh_min=args.soh_min,
        drop_t0=True,
    )
    ds_val = PerovCellTimepoints(
        val_cells,
        channel_stats=ch_stats,
        predict=tuple(args.predict),
        augment=False,
        soh_max=args.soh_max,
        soh_min=args.soh_min,
        drop_t0=True,
    )
    ds_test = PerovCellTimepoints(
        test_cells,
        channel_stats=ch_stats,
        predict=tuple(args.predict),
        augment=False,
        soh_max=args.soh_max,
        soh_min=args.soh_min,
        drop_t0=True,
    )

    # Binning for WeightedRandomSampler
    n_bins = 20
    bins = np.linspace(args.soh_min, args.soh_max, n_bins + 1)

    soh_values = []
    for i in range(len(ds_train)):
        y = ds_train[i]["y"]["soh_avg"]
        if np.isfinite(y):
            soh_values.append(y)
        else:
            soh_values.append(np.nan)

    soh_values = np.array(soh_values)

    valid = np.isfinite(soh_values)
    bin_ids = np.digitize(soh_values[valid], bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_counts = np.bincount(bin_ids, minlength=n_bins)

    weights = np.zeros(len(ds_train), dtype=np.float32)

    for i in range(len(ds_train)):
        y = soh_values[i]
        if not np.isfinite(y):
            weights[i] = 0.0  # never sample invalid targets
        else:
            b = np.digitize(y, bins) - 1
            b = min(max(b, 0), n_bins - 1)
            weights[i] = 1.0 / max(bin_counts[b], 1)

    weights = weights / weights.sum()

    # Summary
    print("Dataset summary:")
    print(f"  Train cells       : {len(train_cells)}")
    print(f"  Validation cells  : {len(val_cells)}")
    print(f"  Test cells        : {len(test_cells)}")
    print(f"  Train samples     : {len(ds_train)}")
    print(f"  Validation samples: {len(ds_val)}")
    print(f"  Test samples      : {len(ds_test)}")

    # Loaders
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(weights),
        replacement=True,
    )

    # print("SoH bin counts (train):")
    # for i in range(n_bins):
    #     print(
    #         f"[{bins[i]:.2f}, {bins[i+1]:.2f}): {bin_counts[i]}"
    #     )

    loaders = {
        "train": DataLoader(
            ds_train,
            batch_size=args.batch_size,
            sampler=sampler,
            # shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            ds_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            ds_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        ),
    }

    visualise_dataloader_soh(
        loaders["train"],
        out_dir=Path(args.out_dir) / f"fold_{fold_id}",
        soh_min=args.soh_min,
        soh_max=args.soh_max,
        n_bins=n_bins,
        max_batches=100,  # enough to see behavior
    )

    splits = {
        "train": train_cells,
        "val": val_cells,
        "test": test_cells,
    }

    splits_serializable = {
        "train": serialize_cells(train_cells),
        "val": serialize_cells(val_cells),
        "test": serialize_cells(test_cells),
    }

    with open(Path(args.out_dir) / f"fold_{fold_id}" / "splits.json", "w") as f:
        json.dump(splits_serializable, f, indent=2)

    # Target stats
    target_stats = {
        "train": collect_true_targets(loaders["train"], args.predict),
        "val": collect_true_targets(loaders["val"], args.predict),
        "test": collect_true_targets(loaders["test"], args.predict),
    }

    return meta, channels, ch_stats, loaders, splits, target_stats


def serialize_cells(cells):
    return sorted([p.name for p in cells])


def finite_by_stack(loader, target):
    c = Counter()
    for batch in loader:
        sc = batch["stack_code"].cpu().numpy()
        y = batch["y"][target].cpu().numpy()
        m = np.isfinite(y)
        for s in np.unique(sc):
            c[int(s)] += int(np.sum(m & (sc == s)))
    return dict(c)


# ----------------- Training -----------------
def train_loop(model, loaders, args, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
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

    best_val = float("inf")
    best_epoch = 0
    wait = 0
    best_metrics_tr = None
    best_metrics_va = None
    log_rows = []

    dl_train = loaders["train"]
    dl_val = loaders["val"]

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0

        for batch in dl_train:
            x = batch["x"].to(device)
            sc = batch["stack_code"].to(device).long()
            y = {k: batch["y"][k].to(device) for k in args.predict}

            out = model(x, sc)
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
            train_loss_sum += loss.item()

        sched.step()
        train_loss_epoch = train_loss_sum / max(len(dl_train), 1)

        # Validation loss
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in dl_val:
                x = batch["x"].to(device)
                sc = batch["stack_code"].to(device).long()
                y = {k: batch["y"][k].to(device) for k in args.predict}
                out = model(x, sc)
                loss = compute_batch_loss(
                    out,
                    y,
                    predict=args.predict,
                    consistency_weight=args.consistency_weight,
                    consistency_log=args.consistency_log,
                )
                val_loss_sum += loss.item()

        val_loss_epoch = val_loss_sum / max(len(dl_val), 1)

        overall_tr, _ = evaluate_regression(
            model, dl_train, device, predict=tuple(args.predict)
        )
        overall_va, _ = evaluate_regression(
            model, dl_val, device, predict=tuple(args.predict)
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss_epoch,
            "val_loss": val_loss_epoch,
        }

        for k in args.predict:
            row[f"train_mae_{k}"] = overall_tr["mae"][k]
            row[f"val_mae_{k}"] = overall_va["mae"][k]
            row[f"train_rmse_{k}"] = overall_tr["rmse"][k]
            row[f"val_rmse_{k}"] = overall_va["rmse"][k]
            row[f"train_r2_{k}"] = overall_tr["r2"][k]
            row[f"val_r2_{k}"] = overall_va["r2"][k]

        log_rows.append(row)

        primary = f"val_mae_{args.predict[0]}"
        if row[primary] < best_val:
            best_val = row[primary]
            best_epoch = epoch
            wait = 0
            best_metrics_tr = overall_tr
            best_metrics_va = overall_va

            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1

        val_str = " ".join([f"{k}:{overall_va['mae'][k]:.4f}" for k in args.predict])
        print(
            f"[{epoch:03d}] loss {row['train_loss']:.4f} | val {val_str} | best {best_val:.4f} "
            f"(epoch {best_epoch:03d})"
        )

        if wait >= args.patience:
            print(f"Early stopping at epoch {epoch}, best was {best_epoch}")
            break

    return model, log_rows, best_epoch, best_metrics_tr, best_metrics_va, best_state


# ----------------- Metrics -----------------
def masked_mae(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Computes MAE while ignoring NaNs."""
    mask = torch.isfinite(true).float()
    return (torch.abs(pred - true) * mask).sum() / mask.sum().clamp_min(1.0)


def evaluate_regression(
    model,
    loader,
    device,
    predict=("soh_avg",),
    per_stack=False,
    return_preds=False,
):
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

    # optional prediction storage
    if return_preds:
        preds_true = {k: [] for k in predict}
        preds_pred = {k: [] for k in predict}
        meta = {
            "cell_file": [],
            "t_local": [],
            "stack_code": [],
        }

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

                if return_preds:
                    preds_true[k].append(y_m.detach().cpu())
                    preds_pred[k].append(pred.detach().cpu())

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

            if return_preds:
                meta["cell_file"].extend(batch["cell_file"])
                meta["t_local"].append(batch["t_local"].detach().cpu())
                meta["stack_code"].append(sc.detach().cpu())

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

    if return_preds:
        preds = {
            "y_true": {k: torch.cat(preds_true[k]).numpy() for k in predict},
            "y_pred": {k: torch.cat(preds_pred[k]).numpy() for k in predict},
            "cell_file": meta["cell_file"],
            "t_local": torch.cat(meta["t_local"]).numpy(),
            "stack_code": torch.cat(meta["stack_code"]).numpy(),
        }

    if return_preds:
        return overall, per_stack_metrics, preds

    return overall, per_stack_metrics


def run_evaluation(model, loaders, args, device):
    """
    Runs evaluation on all available splits.
    Returns metrics AND raw predictions.
    """
    results = {}

    for split, loader in loaders.items():
        if loader is None or len(loader.dataset) == 0:
            continue

        overall, per_stack, preds = evaluate_regression(
            model,
            loader,
            device,
            predict=tuple(args.predict),
            per_stack=True,
            return_preds=True,  # IMPORTANT
        )

        results[split] = {
            "overall": overall,
            "per_stack": per_stack,
            "preds": preds,
        }

    return results


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
            cons = torch.nn.functional.l1_loss(s_hat, rv_hat * rj_hat * rf_hat)

        loss = loss + consistency_weight * cons

    return loss


def aggregate_cv_metrics(cv_records, split, metric, targets):
    """
    cv_records: list of fold records
    split: "train" or "val"
    metric: "mae", "rmse", or "r2"
    targets: list of target names
    """
    out = {}

    for t in targets:
        values = [r[split]["overall"][metric][t] for r in cv_records]
        out[t] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "per_fold": [float(v) for v in values],
        }

    return out


# ----------------- Training -----------------
def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cv_metrics = []

    for k in tqdm(range(args.n_folds), desc="Training with CV"):
        print(f"Training on fold #{k}")
        fold_dir = out_dir / f"fold_{k}"
        fold_dir.mkdir(exist_ok=True)

        meta, channels, ch_stats, loaders, splits, target_stats = (
            build_datasets_and_loaders(args, fold_id=k)
        )

        for split, target in target_stats.items():
            plot_target_histograms(
                target,
                out_dir=fold_dir,
                split=split,
            )

        # Save channel stats
        np.savez(fold_dir / "channel_stats.npz", **ch_stats)

        # Model
        all_stacks = set()
        for cells in splits.values():
            for cf in cells:
                dat = np.load(cf, allow_pickle=True)
                all_stacks.add(
                    int(dat["stack_code"]) if "stack_code" in dat.files else 0
                )
        n_stacks = max(all_stacks) + 1 if len(all_stacks) > 0 else 1

        model = BaselineMLP(
            n_stacks=n_stacks,
            in_ch=len(channels),
            predict=tuple(args.predict),
            use_stack=args.use_stack,
            d_emb=2,
            hidden=(64, 32),
            dropout=0.0,
        ).to(device)

        # Train
        model, log_rows, best_epoch, best_tr, best_va, best_state = train_loop(
            model, loaders, args, device
        )

        # IMPORTANT: restore best model before evaluation
        model.load_state_dict(best_state)

        # Save checkpoint
        torch.save(
            {
                "model": model.state_dict(),
                "meta": meta,
                "channel_stats": ch_stats,
                "predict": list(args.predict),
            },
            fold_dir / "model.pt",
        )

        print(f"Done. Best model saved to {fold_dir/'model.pt'}")

        # Logs
        plot_loss_history(log_rows, fold_dir / "loss_history.png")

        # Evaluation
        eval_results = run_evaluation(model, loaders, args, device)

        fold_record = {
            "fold": k,
            "best_epoch": best_epoch,
            "train": {
                "overall": eval_results["train"]["overall"],
            },
            "val": {
                "overall": eval_results["val"]["overall"],
            },
            "test": {
                "overall": eval_results["test"]["overall"],
            },
        }
        all_cv_metrics.append(fold_record)

        for split in ["train", "val", "test"]:
            if split not in eval_results:
                continue

            preds = eval_results[split]["preds"]

            split_dir = fold_dir / f"parity_{split}"
            split_dir.mkdir(exist_ok=True)

            parity_plots_all(
                preds,
                predict=args.predict,
                out_dir=split_dir,
                prefix=f"parity_{split}_fold_{k}",
            )

            print(f"[Fold {k}] Saved {split} parity plots to {split_dir}")

    cv_summary = {
        "train": {
            "mae": aggregate_cv_metrics(all_cv_metrics, "train", "mae", args.predict),
            "rmse": aggregate_cv_metrics(all_cv_metrics, "train", "rmse", args.predict),
            "r2": aggregate_cv_metrics(all_cv_metrics, "train", "r2", args.predict),
        },
        "val": {
            "mae": aggregate_cv_metrics(all_cv_metrics, "val", "mae", args.predict),
            "rmse": aggregate_cv_metrics(all_cv_metrics, "val", "rmse", args.predict),
            "r2": aggregate_cv_metrics(all_cv_metrics, "val", "r2", args.predict),
        },
        "test": {
            "mae": aggregate_cv_metrics(all_cv_metrics, "test", "mae", args.predict),
            "rmse": aggregate_cv_metrics(all_cv_metrics, "test", "rmse", args.predict),
            "r2": aggregate_cv_metrics(all_cv_metrics, "test", "r2", args.predict),
        },
    }

    config = {
        "n_folds": args.n_folds,
        "test_split": args.test_split,
        "predict": list(args.predict),
        "soh_min": args.soh_min,
        "soh_max": args.soh_max,
        "learning_rate": args.lr,
        "weight_decay": args.wd,
        "consistency_weight": args.consistency_weight,
        "consistency_log": args.consistency_log,
        "seed": args.seed,
    }

    metrics_out = {
        "cv_per_fold": all_cv_metrics,
        "cv_summary": cv_summary,
        "config": config,
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_out, indent=2),
        encoding="utf-8",
    )

    return model, loaders


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
        "--n-folds",
        type=int,
        default=1,
        help="Number of CV folds. 1 = single split (default).",
    )
    ap.add_argument("--soh-max", type=float, default=None)
    ap.add_argument("--soh-min", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--use-stack", default=False, action="store_true")
    ap.add_argument("--drop-stack", dest="use-stack", action="store_false")
    # ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--test-split", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-aug", action="store_true")

    ap.add_argument("--consistency-weight", type=float, default=0.0)
    ap.add_argument("--consistency-log", action="store_true")

    args = ap.parse_args()

    # ----------------- Resolve data roots -----------------
    if args.data_roots is not None and args.data_parent is not None:
        raise ValueError("Provide either --data-roots OR --data-parent, not both.")

    if args.data_roots is None and args.data_parent is None:
        raise ValueError("You must provide either --data-roots or --data-parent.")

    if args.data_parent is not None:
        parent = Path(args.data_parent)
        if not parent.exists():
            raise FileNotFoundError(f"--data-parent does not exist: {parent}")

        # dataset roots are subfolders containing meta.json
        roots = []
        for p in sorted(parent.iterdir()):
            if p.is_dir() and (p / "meta.json").exists():
                roots.append(str(p))

        if len(roots) == 0:
            raise RuntimeError(
                f"No dataset roots found in {parent}. "
                "Expected subfolders containing meta.json."
            )

        args.data_roots = roots
        print(f"Discovered {len(roots)} dataset roots under {parent}")
    else:
        # normalize paths to strings
        args.data_roots = [str(Path(p)) for p in args.data_roots]
        # quick sanity check
        for r in args.data_roots:
            if not (Path(r) / "meta.json").exists():
                raise FileNotFoundError(f"meta.json not found in dataset root: {r}")

    # ----------------- Run training -----------------
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
  --test-split 0.1 \
  --seed 42 \
  --consistency-weight 0.5 \
  --consistency-log
"""
