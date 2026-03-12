import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap


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
    ax.set_yscale("log")
    ax.set_title("Training loss history")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ----------------- Parity -----------------
def parity_plot_from_preds(preds, target: str, out_path: Path):
    """Save a parity (y_true vs y_pred) scatter for a given target from precomputed preds."""
    if "y_true" not in preds or "y_pred" not in preds:
        raise ValueError("preds must contain 'y_true' and 'y_pred' dicts.")

    if target not in preds["y_true"] or target not in preds["y_pred"]:
        raise KeyError(f"Target '{target}' not found in preds.")

    y_true = np.asarray(preds["y_true"][target])
    y_pred = np.asarray(preds["y_pred"][target])

    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        print(f"[WARN] No finite '{target}' values for parity plot.")
        return

    y_true = y_true[m]
    y_pred = y_pred[m]

    fig, ax = plt.subplots(figsize=(4, 4), dpi=160)
    ax.scatter(y_true, y_pred, s=6, alpha=0.6, c="#388088")

    lo = float(np.min([y_true.min(), y_pred.min()]))
    hi = float(np.max([y_true.max(), y_pred.max()]))
    pad = 0.02 * (hi - lo + 1e-8)

    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], lw=1.0, c="#c33033")
    ax.set_xlabel(f"True {target}")
    ax.set_ylabel(f"Pred {target}")
    ax.set_title(f"Parity: {target}")
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parity_plots_all(preds, predict, out_dir: Path, prefix="parity"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in predict:
        out_path = out_dir / f"{prefix}_{k}.png"
        parity_plot_from_preds(preds, target=k, out_path=out_path)


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
            y, bins=bins, density=True, alpha=0.8, edgecolor="black", color="#388088"
        )

        ax.set_xlabel(k)
        ax.set_ylabel("Density")
        ax.set_title(f"{k} distribution ({split})")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{split}_{k}.png")
        plt.close(fig)


def visualise_dataloader_soh(
    dl,
    out_dir: Path,
    soh_min=0.8,
    soh_max=1.2,
    n_bins=10,
    max_batches=None,
    with_plot=True,
):
    """
    Visualise SoH-bin composition of batches from a DataLoader.

    Args:
        dl: DataLoader
        soh_min, soh_max: SoH range
        n_bins: number of SoH bins
        max_batches: limit number of batches to inspect (None = all)
        with_plot: whether to produce matplotlib plots
    """
    bins = np.linspace(soh_min, soh_max, n_bins + 1)

    batch_bin_counts = []
    idxs_seen = []

    for i, batch in enumerate(dl):
        if max_batches is not None and i >= max_batches:
            break

        y = batch["y"]["soh_avg"].cpu().numpy()
        m = np.isfinite(y)
        y = y[m]

        bin_ids = np.digitize(y, bins) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)

        counts = np.bincount(bin_ids, minlength=n_bins)
        batch_bin_counts.append(counts)

        # track sample indices if available
        if "idx" in batch:
            idxs_seen.extend(batch["idx"].tolist())

    batch_bin_counts = np.array(batch_bin_counts)

    if with_plot:
        out_path = out_dir / "dl_train_balance.png"

        fig, ax = plt.subplots(figsize=(14, 6))

        for b in range(n_bins):
            ax.plot(
                batch_bin_counts[:, b],
                label=f"[{bins[b]:.2f}, {bins[b+1]:.2f})",
                alpha=0.8,
            )

        ax.set_xlabel("Batch index")
        ax.set_ylabel("Samples per batch")
        ax.set_title(r"$R_\mathrm{PCE}$-bin composition per batch")
        ax.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        plt.savefig(out_path)

        # mean_counts = batch_bin_counts.mean(axis=0)
        # print("Average samples per batch per SoH bin:")
        # for b in range(n_bins):
        #     print(
        #         f"  [{bins[b]:.2f}, {bins[b+1]:.2f}): {mean_counts[b]:.2f}"
        #     )

    return batch_bin_counts, idxs_seen


# ----------------- Trajectories -----------------
def plot_ensemble_trajectory(cell_data, cell_file, out_path):
    d = cell_data[cell_file]

    t = d["t"]
    y_true = d["y_true"]
    y_mean = d["y_pred_mean"]
    y_std = d["y_pred_std"]

    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=160)

    # True trajectory
    ax.plot(t, y_true, "-o", label="True", color="#f88088", linewidth=1.8)

    # Ensemble mean
    ax.plot(t, y_mean, "-o", label="Ensemble mean", color="#088088", linewidth=1.8)

    # Uncertainty shading (mean ± std)
    ax.fill_between(
        t,
        y_mean - y_std,
        y_mean + y_std,
        color="#088088",
        alpha=0.2,
        label="±1 std (across folds)",
    )

    ax.set_xlabel("Time index")
    ax.set_ylabel(r"$R_\mathrm{PCE}$")
    ax.set_ylim(0.8, 1.2)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_multiple_ensemble_trajectories(
    cell_data,
    cell_files,
    out_path,
    y_range=(0.8, 1.2),
):
    """
    Plot multiple ensemble trajectories in a single multi-panel figure.

    Args:
        cell_data: dict returned by collect_ensemble_timeseries
        cell_files: list of 3 cell file paths (strings)
        out_path: where to save figure
    """

    n = len(cell_files)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5), dpi=300, sharey=True)

    panel_labels = ["(a)", "(b)", "(c)"]

    if n == 1:
        axes = [axes]

    for i, (ax, cf) in enumerate(zip(axes, cell_files)):
        d = cell_data[cf]

        t = d["t"]
        y_true = d["y_true"]
        y_mean = d["y_pred_mean"]
        y_std = d["y_pred_std"]

        # True
        ax.plot(t, y_true, "-o", color="#f88088", linewidth=1.8, label="True")

        # Mean
        ax.plot(
            t,
            y_mean,
            "-o",
            color="#088088",
            linewidth=1.8,
            label="Ensemble mean",
        )

        # Shading
        ax.fill_between(
            t,
            y_mean - y_std,
            y_mean + y_std,
            color="#088088",
            alpha=0.2,
        )

        ax.text(
            0.02,
            0.96,
            panel_labels[i],
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
        )

        # ax.set_title(Path(cf).stem)
        ax.set_xlabel("Elapsed time (hours)")
        ax.set_ylim(*y_range)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"$R_\mathrm{PCE}$")

    # single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path)
    fig.savefig(str(out_path).replace("png", "pdf"))
    plt.close(fig)


def plot_absolute_error_vs_time(
    t_hours,
    abs_err,
    out_path: Path,
    gridsize: int = 35,
    mincnt: int = 2,
    ylabel: str = r"Absolute error $|\hat{y} - y|$",
    inset_xlim=(0.0, 15.0),
    inset_ylim=(0.0, 0.25),
    inset_gridsize: int = 22,
    inset_mincnt: int = 1,
):
    """
    Hexbin plot of absolute error vs elapsed time, with an inset zoom for early time.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    t_hours = np.asarray(t_hours, dtype=float)
    abs_err = np.asarray(abs_err, dtype=float)

    m = np.isfinite(t_hours) & np.isfinite(abs_err) & (t_hours > 0)
    t_hours = t_hours[m]
    abs_err = abs_err[m]

    if t_hours.size == 0:
        print("[WARN] No finite values for error vs time plot.")
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=180)
    colors_hi_contrast = ["#004050", "#088088", "#38b8b0", "#f8c088", "#f88088"]
    cmap_teal_salmon_hi = LinearSegmentedColormap.from_list(
        "teal_salmon_hi", colors_hi_contrast, N=256
    )

    hb = ax.hexbin(
        t_hours,
        abs_err,
        gridsize=gridsize,
        cmap=cmap_teal_salmon_hi,
        mincnt=mincnt,
        # bins="log",
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Count")

    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.20)
    ax.set_axisbelow(True)

    # -----------------------------
    # Inset: early-time zoom
    # -----------------------------
    axins = inset_axes(
        ax,
        width="42%",  # tweak if you want it larger/smaller
        height="42%",
        loc="upper right",
        borderpad=1.0,
    )

    # restrict data to inset region for better binning
    x0, x1 = inset_xlim
    y0, y1 = inset_ylim
    mi = (t_hours >= x0) & (t_hours <= x1) & (abs_err >= y0) & (abs_err <= y1)

    axins.hexbin(
        t_hours[mi],
        abs_err[mi],
        gridsize=inset_gridsize,
        cmap=cmap_teal_salmon_hi,
        mincnt=inset_mincnt,
        # bins="log",
    )

    axins.set_xlim(x0, x1)
    axins.set_ylim(y0, y1)
    axins.grid(True, alpha=0.15)
    axins.set_axisbelow(True)
    axins.set_xticks([0, 5, 10, 15])
    axins.set_yticks([0.0, 0.1, 0.2])

    # draw a rectangle on the main plot showing inset region
    ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.4, lw=0.8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
