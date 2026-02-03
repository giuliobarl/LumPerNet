import random
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
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

# ----------------- Loss history -----------------
def plot_loss_history(log_rows, out_path: Path):
    epochs = [r["epoch"] for r in log_rows]
    train_loss = [r["train_loss"] for r in log_rows]
    val_loss = [r["val_loss"] for r in log_rows]

    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=160)
    ax.plot(epochs, train_loss, label="Train loss", lw=2, color="#388088")
    ax.plot(epochs, val_loss, label="Validation loss", lw=2, color="#c33033")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss history")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# ----------------- Parity -----------------
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

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(yh)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=160)
    ax.scatter(y_true, y_pred, s=6, alpha=0.6, color="#388088")
    lo, hi = float(np.nanmin([y_true.min(), y_pred.min()])), float(
        np.nanmax([y_true.max(), y_pred.max()])
    )
    pad = 0.02 * (hi - lo + 1e-8)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], lw=1.0, color="#c33033")
    ax.set_xlabel(f"True {target}")
    ax.set_ylabel(f"Pred {target}")
    ax.set_title(f"Parity: {target}")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def parity_plots_all(model, loader, device, targets, out_dir: Path, prefix="parity_val"):
    for k in targets:
        out_path = out_dir / f"{prefix}_{k}.png"
        parity_plot(
            model,
            loader,
            device,
            target=k,
            out_path=out_path,
        )

# ----------------- Target stats -----------------
def plot_target_histograms(
    data,
    out_dir: Path,
    split: str,
    bins=50,
):
    """
    data: dict[target] -> np.ndarray of true y
    split: "train" or "val"
    """
    for k, y in data.items():
        if y is None:
            print(f"[WARN] No valid values for {k} ({split})")
            continue

        fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=160)

        ax.hist(
            y,
            bins=bins,
            density=True,
            alpha=0.8,
            edgecolor="black",
            color="#388088"
        )

        ax.set_xlabel(k)
        ax.set_ylabel("Density")
        ax.set_title(f"{k} distribution ({split})")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{split}_{k}.png")
        plt.close(fig)

# ----------------- Trajectories -----------------
def collect_cell_timeseries(
    model,
    loader,
    device,
    target="soh_avg",
):
    """
    Collects true and predicted target values per cell and per time index.

    Returns:
        dict[cell_file] -> {
            "t": np.ndarray,
            "y_true": np.ndarray,
            "y_pred": np.ndarray,
        }
    """
    model.eval()
    data = defaultdict(lambda: {"t": [], "y_true": [], "y_pred": []})

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            sc = batch["stack_code"].to(device).long()
            t_local = batch["t_local"].cpu().numpy()
            cell_files = batch["cell_file"]

            y_true = batch["y"][target].to(device)
            y_pred = model(x, sc)[target]

            for i in range(len(cell_files)):
                yt = y_true[i].item()
                if not np.isfinite(yt):
                    continue

                cf = cell_files[i]
                data[cf]["t"].append(t_local[i])
                data[cf]["y_true"].append(yt)
                data[cf]["y_pred"].append(y_pred[i].item())

    # convert to sorted numpy arrays
    out = {}
    for cf, d in data.items():
        t = np.array(d["t"])
        idx = np.argsort(t)
        out[cf] = {
            "t": t[idx],
            "y_true": np.array(d["y_true"])[idx],
            "y_pred": np.array(d["y_pred"])[idx],
        }

    return out

def plot_cell_trajectory(
    cell_data,
    cell_file,
    out_path,
    target="soh_avg",
    y_range=(0.8, 1.2),
):
    """
    Plots true vs predicted target trajectory for one cell.
    """
    # out_path.mkdir(parents=True, exist_ok=True)
    d = cell_data[cell_file]

    t = d["t"]
    y_true = d["y_true"]
    y_pred = d["y_pred"]

    # restrict to validity window
    m = (y_true >= y_range[0]) & (y_true <= y_range[1])
    if m.sum() < 2:
        print(f"[WARN] Not enough points in range for {cell_file}")
        return

    t = t[m]
    y_true = y_true[m]
    y_pred = y_pred[m]

    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=160)

    ax.plot(t, y_true, "-o", label="True", lw=2, markersize=4, color="#388088")
    ax.plot(t, y_pred, "--o", label="Predicted", lw=2, markersize=4, color="#c33033")

    ax.set_xlabel("Time index")
    ax.set_ylabel(target)
    ax.set_title(f"{target} trajectory\n{cell_file}")
    ax.set_ylim(*y_range)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_representative_cells(
    model,
    loader,
    device,
    cell_files,
    out_dir,
    target="soh_avg",
    y_range=(0.8, 1.2),
    prefix="trajectory",
):
    """
    Plots trajectories for a list of cell_files.
    """
    cell_data = collect_cell_timeseries(
        model,
        loader,
        device,
        target=target,
    )

    for cf in cell_files:
        if cf not in cell_data:
            print(f"[WARN] Cell {cf} not found in loader")
            continue

        out_path = (
            out_dir / f"{prefix}_{target}_{cf.split('\\')[-1].replace('.npz','')}.png"
        )

        print(out_path)

        plot_cell_trajectory(
            cell_data,
            cell_file=cf,
            out_path=out_path,
            target=target,
            y_range=y_range,
        )
