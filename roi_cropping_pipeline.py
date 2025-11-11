"""
ROI cropping & preprocessing pipeline for EL/PL image series (interactive).

New in this version:
- Custom naming via --grid-names "comma,separated,names,in,row-major,order"
- Preset naming via --preset chose4x4 (maps to the user-provided 4x4 layout)
- Prints the final (name, cx, cy) mapping after selection/import

Author: ChatGPT
Dependencies:
  - Python 3.8+
  - numpy, matplotlib, imageio (or tifffile), opencv-python (optional), pillow, pandas, scipy
Install (example):
  pip install numpy matplotlib imageio tifffile pillow pandas scipy opencv-python

Usage (interactive ROI selection then batch crop):
  python roi_cropping_pipeline.py \
      --ref-image /path/to/reference_PL_image.tif \
      --images-root /path/to/experiment_root \
      --modalities EL PLoc PLsc \
      --output-root /path/to/output_crops \
      --crop-size 56 \
      --rows 4 --cols 4 \
      --preset chose4x4 \
      --hot-pixels true \
      --dark-frame /path/to/dark_frame.tif \
      --flat-field /path/to/flat_field.tif

If you already have ROI centers (csv), you can skip the interactive step:
  python roi_cropping_pipeline.py --ref-image ... --images-root ... --modalities ... \
      --roi-csv /path/to/roi_centers.csv --output-root ... --preset chose4x4

roi_centers.csv format:
  roi_id, cx, cy
  ch01, 123.0, 456.0

Outputs:
  output_root/
    <modality>/<channel_name>/t_####.tif
  output_root/masks/<channel_name>_mask.png
  output_root/manifest.csv (metadata of every saved crop)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from tqdm import tqdm

# Prefer tifffile if available (16-bit safe), else fall back to imageio
try:
    import tifffile as tifi

    def imread(path):
        return tifi.imread(str(path))

    def imwrite(path, arr):
        tifi.imwrite(str(path), arr)

except Exception:
    import imageio.v2 as imageio

    def imread(path):
        return imageio.imread(path)

    def imwrite(path, arr):
        imageio.imwrite(path, arr)


matplotlib.use("Agg")  # non-interactive by default; switching to TkAgg for picking

# ----------------------- Custom naming helpers -----------------------


def parse_grid_names(grid_names_str: str | None, preset: str | None, n: int | None):
    """
    Returns a list of names or None.
    - grid_names_str: comma-separated names in row-major order.
    - preset: currently supports "chose4x4".
    - n: expected length (optional check).
    """
    names = None
    if preset:
        if preset.lower() == "chose4x4":
            names = [
                "2B",
                "2C",
                "1C",
                "1B",
                "2A",
                "2D",
                "1D",
                "1A",
                "3B",
                "3C",
                "4B",
                "4C",
                "3A",
                "3D",
                "4A",
                "4D",
            ]
        elif preset.lower() == "chose3x4":
            names = [
                "2B",
                "2C",
                "1C",
                "1B",
                "2A",
                "2D",
                "1D",
                "1A",
                "3B",
                "3C",
                "3A",
                "3D",
            ]
        else:
            print(f"WARNING: Unknown preset '{preset}', ignoring.")
    if grid_names_str:
        names = [s.strip() for s in grid_names_str.split(",") if s.strip()]
    if names is not None and n is not None:
        if len(names) >= n:
            names = names[:n]  # truncate to number of selected ROIs
        else:
            print(
                f"WARNING: Provided {len(names)} names but {n} ROIs were selected; ignoring custom names."
            )
            return None

    return names


# ----------------------- Utilities -----------------------


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_image_gray(path: Path) -> np.ndarray:
    img = imread(path)
    if img.ndim == 3:
        img = img[..., 0] if img.shape[-1] == 1 else np.mean(img, axis=-1)
    return img.astype(np.float32)


def apply_dark_flat(
    img: np.ndarray, dark: np.ndarray = None, flat: np.ndarray = None, eps: float = 1e-6
) -> np.ndarray:
    """
    Correct image with dark subtraction and flat-field normalization:
      img_corr = (img - dark) / (flat - dark)
    Then normalize to ~0..1 (99.9th percentile) for float outputs.
    """
    out = img.copy()
    if dark is not None:
        out = out - dark
    if flat is not None:
        denom = flat - (dark if dark is not None else 0.0)
        denom = np.where(np.abs(denom) < eps, eps, denom)
        out = out / denom
        # out = out / (np.percentile(out, 99.9) + eps)
    return out


def remove_hot_pixels(
    img: np.ndarray,
    ksize: int = 3,
    thresh_sigma: float = 10.0,
    sigma_floor: float = 3.0,
):
    """
    Conservative hot-pixel removal:
      - only positive spikes (not negative dips)
      - pixel must be a local maximum
      - threshold = max(MAD-based sigma, sigma_floor)
    """
    from scipy.ndimage import maximum_filter, median_filter

    med = median_filter(img, size=ksize, mode="reflect")
    resid = img - med  # positive -> spike

    # robust sigma via MAD, with a floor to avoid over-flagging when the image darkens
    mad = np.median(np.abs(resid - np.median(resid)))
    sigma = 1.4826 * mad
    sigma = max(sigma, float(sigma_floor))

    # candidate spikes: positive only
    cand = resid > (thresh_sigma * sigma)

    # keep only local maxima to avoid "texture smoothing"
    local_max = img >= maximum_filter(img, size=3, mode="reflect")
    mask = cand & local_max

    # Winsorize (clip) rather than replace by full median to reduce visual flattening
    out = img.copy()
    out[mask] = med[mask] + (thresh_sigma * sigma)

    # print(f"hot-pixel replace ratio: {mask.mean()*100:.4f}%")
    return out


def clamp_center(cx, cy, half, w, h):
    cx = int(np.clip(cx, half, w - half - 1))
    cy = int(np.clip(cy, half, h - half - 1))
    return cx, cy


def crop_center(img: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    half = size // 2
    xs, xe = cx - half, cx + half
    ys, ye = cy - half, cy + half
    return img[ys:ye, xs:xe]


def sort_row_major(centers, rows=None, cols=None, names_override=None):
    """
    Robust row-major ordering:
      - auto-detect rows by y-gaps (works with incomplete last rows)
      - sort each row by x
      - then assign names in row-major order

    If names_override is given, it will be truncated to N points.
    """
    import numpy as np

    pts = np.array(centers, dtype=float)  # (N,2) = [cx, cy]
    N = len(centers)
    if N == 0:
        return []

    # names
    if names_override is None:
        names = [f"ch{(i+1):02d}" for i in range(N)]
    else:
        names = list(names_override[:N])

    # 1) sort by y (top→bottom)
    idx_y = np.argsort(pts[:, 1])
    py = pts[idx_y]

    # 2) detect row breaks by y-gaps (adaptive threshold)
    dy = np.diff(py[:, 1])
    # robust threshold: 2× median intra-row spacing (tweak if needed)
    thr = 2.0 * (np.median(dy[dy > 0]) if np.any(dy > 0) else 1.0)

    breaks = [0]
    for i, d in enumerate(dy, start=1):
        if d > thr:
            breaks.append(i)
    breaks.append(N)

    # 3) within each row, sort by x (left→right)
    ordered = []
    for a, b in zip(breaks[:-1], breaks[1:]):
        row = py[a:b]
        row = row[np.argsort(row[:, 0])]
        ordered.append(row)

    ordered = np.vstack(ordered) if ordered else np.empty((0, 2))
    return [
        (names[i], int(ordered[i, 0]), int(ordered[i, 1]))
        for i in range(ordered.shape[0])
    ]


def interactive_pick_boxes(image_path: Path, n_rois: int = None, crop_size: int = 56):
    """
    Interactive ROI selection (two clicks per ROI: top-left, bottom-right).
    Returns list of center points [(cx,cy), ...].
    """
    matplotlib.use("TkAgg", force=True)

    img = load_image_gray(image_path)
    h, w = img.shape

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap="gray")
    ax.set_title(
        "Click TOP-LEFT then BOTTOM-RIGHT for each ROI (close window when done)"
    )
    plt.tight_layout()

    boxes = []
    rects = []

    def onclick(event):
        nonlocal boxes, rects
        if event.inaxes != ax:
            return
        x, y = int(event.xdata), int(event.ydata)
        if len(boxes) % 2 == 0:
            boxes.append((x, y))
            ax.plot(x, y, "r+")
            fig.canvas.draw()
        else:
            boxes.append((x, y))
            x0, y0 = boxes[-2]
            x1, y1 = boxes[-1]
            xs, ys = min(x0, x1), min(y0, y1)
            xe, ye = max(x0, x1), max(y0, y1)
            rect = plt.Rectangle(
                (xs, ys), xe - xs, ye - ys, fill=False, edgecolor="r", linewidth=1.0
            )
            ax.add_patch(rect)
            rects.append(rect)
            cx = (xs + xe) // 2
            cy = (ys + ye) // 2
            half = crop_size // 2
            cx, cy = clamp_center(cx, cy, half, w, h)
            preview = plt.Rectangle(
                (cx - half, cy - half),
                crop_size,
                crop_size,
                fill=False,
                edgecolor="y",
                linewidth=1.2,
                linestyle="--",
            )
            ax.add_patch(preview)
            fig.canvas.draw()

        if n_rois is not None and len(boxes) // 2 >= n_rois:
            fig.canvas.mpl_disconnect(cid)

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    centers = []
    for i in range(0, len(boxes), 2):
        if i + 1 >= len(boxes):
            break
        (x0, y0), (x1, y1) = boxes[i], boxes[i + 1]
        xs, ys = min(x0, x1), min(y0, y1)
        xe, ye = max(x0, x1), max(y0, y1)
        cx = (xs + xe) // 2
        cy = (ys + ye) // 2
        centers.append((cx, cy))
    return centers


def find_images(images_root: Path, modalities: list[str]) -> dict:
    modality_paths = {}
    for m in modalities:
        paths = sorted(
            [
                p
                for p in images_root.rglob("*")
                if p.is_file()
                and m in p.parts
                and p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg")
            ]
        )
        modality_paths[m] = paths
    return modality_paths


def save_mask(
    output_root: Path, name: str, cx: int, cy: int, size: int, full_h: int, full_w: int
):
    mask = np.zeros((full_h, full_w), dtype=np.uint8)
    half = size // 2
    xs, xe = cx - half, cx + half
    ys, ye = cy - half, cy + half
    xs = max(xs, 0)
    ys = max(ys, 0)
    xe = min(xe, full_w)
    ye = min(ye, full_h)
    mask[ys:ye, xs:xe] = 255
    ensure_dir(output_root / "masks")
    imwrite(output_root / "masks" / f"{name}_mask.png", mask)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive ROI cropping & preprocessing for EL/PL time series."
    )
    parser.add_argument(
        "--ref-image",
        type=str,
        required=True,
        help="Path to a reference PL image used for ROI selection.",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        required=True,
        help="Root folder that contains modality folders with images.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=["EL", "PLoc", "PLsc"],
        help="List of modality folder names to process.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Where to write cropped images and metadata.",
    )
    parser.add_argument(
        "--roi-csv",
        type=str,
        help="Optional CSV with columns [roi_id,cx,cy]. Skips interactive selection.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=56,
        help="Crop width/height in pixels (must be even).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Optional expected grid rows (for naming).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Optional expected grid cols (for naming).",
    )
    parser.add_argument(
        "--grid-names",
        type=str,
        default=None,
        help="Comma-separated channel names in row-major order (length must equal ROI count).",
    )
    parser.add_argument(
        "--preset", type=str, default=None, help="Preset channel map (e.g., chose4x4)."
    )
    parser.add_argument(
        "--hot-pixels",
        type=str,
        default="true",
        help="Apply hot pixel removal (true/false).",
    )
    parser.add_argument(
        "--dark-frame",
        type=str,
        default=None,
        help="Optional dark frame path for correction.",
    )
    parser.add_argument(
        "--dark-map",
        type=str,
        default=None,
        help='Per-modality dark frames, e.g. "EL=/path/dark_EL.tif,PLoc=/path/dark_PLoc.tif,PLsc=/path/dark_PLsc.tif".',
    )
    parser.add_argument(
        "--flat-field",
        type=str,
        default=None,
        help="Optional flat field path for correction.",
    )
    parser.add_argument(
        "--median-ksize",
        type=int,
        default=3,
        help="Median filter size for hot pixel removal.",
    )
    parser.add_argument(
        "--outlier-sigma",
        type=float,
        default=10.0,
        help="Sigma threshold for outliers.",
    )
    parser.add_argument(
        "--sigma-floor",
        type=float,
        default=3.0,
        help="Absolute noise floor (ADU) for despiking.",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable tqdm progress bars."
    )
    args = parser.parse_args()

    if args.crop_size % 2 != 0:
        print("ERROR: --crop-size must be an even number (e.g., 56).")
        sys.exit(1)

    images_root = Path(args.images_root)
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    # Load reference image
    ref_image_path = Path(args.ref_image)
    ref_img = load_image_gray(ref_image_path)
    H, W = ref_img.shape

    dark = load_image_gray(Path(args.dark_frame)) if args.dark_frame else None
    flat = load_image_gray(Path(args.flat_field)) if args.flat_field else None

    # once per session (after loading dark with same preprocessing)
    hot_mask_static = None
    if dark is not None:
        dmed = median_filter(dark, size=3)
        dres = dark - dmed
        d_sigma = 1.4826 * np.median(np.abs(dres - np.median(dres)))
        hot_mask_static = dres > (8 * d_sigma)

    # Optional per-modality darks: --dark-map "EL=...,PLoc=...,PLsc=..."
    dark_by_mod = {}
    hot_mask_by_mod = {}

    if args.dark_map:
        for tok in args.dark_map.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "=" not in tok:
                print(f"WARNING: skipping malformed entry in --dark-map: '{tok}'")
                continue
            key, val = tok.split("=", 1)
            key = key.strip()
            val = val.strip()
            try:
                dimg = load_image_gray(Path(val))
                dark_by_mod[key] = dimg
                # build a per-modality persistent hot-pixel mask
                dmed = median_filter(dimg, size=3)
                dres = dimg - dmed
                d_sigma = 1.4826 * np.median(np.abs(dres - np.median(dres)))
                hot_mask_by_mod[key] = dres > (8 * d_sigma)
            except Exception as e:
                print(
                    f"WARNING: failed to load dark for modality '{key}' from '{val}': {e}"
                )

    # ROI centers
    if args.roi_csv:
        df = pd.read_csv(args.roi_csv)
        centers = [(float(cx), float(cy)) for cx, cy in zip(df["cx"], df["cy"])]
        names = list(df["roi_id"].astype(str).values)
        half = args.crop_size // 2
        centers = [clamp_center(cx, cy, half, W, H) for (cx, cy) in centers]
        # Optional re-name using custom names in row-major order
        custom_names = parse_grid_names(args.grid_names, args.preset, n=len(centers))
        if custom_names is not None:
            ordered = sort_row_major(
                centers, rows=args.rows, cols=args.cols, names_override=custom_names
            )
            roi_list = ordered
        else:
            roi_list = list(
                zip(names, [int(c[0]) for c in centers], [int(c[1]) for c in centers])
            )
    else:
        centers = interactive_pick_boxes(
            ref_image_path, n_rois=None, crop_size=args.crop_size
        )
        if len(centers) == 0:
            print("No ROIs selected. Exiting.")
            sys.exit(0)
        half = args.crop_size // 2
        centers = [clamp_center(cx, cy, half, W, H) for (cx, cy) in centers]
        custom_names = parse_grid_names(args.grid_names, args.preset, n=len(centers))
        ordered = sort_row_major(
            centers, rows=args.rows, cols=args.cols, names_override=custom_names
        )
        roi_list = ordered

    print("Final ROI mapping (name, cx, cy):")
    for rid, cx, cy in roi_list:
        print(f"  {rid:>3s} -> ({cx:4d}, {cy:4d})")

    # Persist ROI centers
    roi_meta = [
        {"roi_id": rid, "cx": int(cx), "cy": int(cy)} for (rid, cx, cy) in roi_list
    ]
    with open(output_root / "roi_centers.json", "w") as f:
        json.dump(
            {
                "image": str(ref_image_path),
                "crop_size": args.crop_size,
                "centers": roi_meta,
            },
            f,
            indent=2,
        )

    # Save masks for visualization
    for rid, cx, cy in roi_list:
        save_mask(output_root, rid, cx, cy, args.crop_size, H, W)

    # Discover images
    mods = [m.strip() for m in args.modalities]
    images = find_images(images_root, mods)
    total_found = sum(len(v) for v in images.values())
    if total_found == 0:
        print("WARNING: Found no images under images_root for the given modalities.")
    else:
        print(
            "Discovered images: " + ", ".join([f"{m}:{len(images[m])}" for m in mods])
        )
    global_total = sum(len(images[m]) for m in mods)

    use_pbar = not args.no_progress
    if use_pbar:
        pbar = tqdm(total=global_total, desc="Cropping & saving", unit="img")

    # Process
    manifest_rows = []
    for m in mods:
        out_mod_dir = output_root / m
        ensure_dir(out_mod_dir)
        for img_path in tqdm(images[m], desc=f"{m}", unit="img", leave=False):
            try:
                img = load_image_gray(img_path)
                # Pick per-modality dark if available, else fall back to global
                d_use = dark_by_mod.get(m, dark)
                img = (
                    apply_dark_flat(img, dark=d_use, flat=flat)
                    if (d_use is not None or flat is not None)
                    else img
                )
                # Apply persistent hot-pixel mask (per modality if available)
                hm = hot_mask_by_mod.get(m, hot_mask_static)
                if hm is not None:
                    img[hm] = median_filter(img, size=3)[hm]

                if str(args.hot_pixels).lower() in ("1", "true", "yes", "y"):
                    img = remove_hot_pixels(
                        img, ksize=args.median_ksize, thresh_sigma=args.outlier_sigma
                    )
                for rid, cx, cy in roi_list:
                    crop = crop_center(img, cx, cy, args.crop_size)
                    stem = img_path.stem  # e.g., MySample_0007
                    m = re.match(r"^(.*)_(\d+)$", stem)
                    name, t = (m.group(1), m.group(2)) if m else (stem, stem)

                    target_dir = out_mod_dir / f"{name}_{rid}"  # modality/name_channel/
                    ensure_dir(target_dir)
                    out_path = target_dir / f"{t}.tif"  # modality/name_channel/t.tif
                    arr = crop

                    # preserve physical scale; clip negatives after dark subtraction
                    arr = np.clip(arr, 0, None)

                    # If array is already uint16, write as-is; otherwise cast with a **fixed** global scale
                    if arr.dtype == np.uint16:
                        imwrite(out_path, arr)
                    else:
                        # choose a fixed scale per run; simplest is assume input is 0..65535 after calibration
                        arr16 = np.clip(arr, 0, 65535).astype(np.uint16)
                        imwrite(out_path, arr16)

                    manifest_rows.append(
                        {
                            "modality": m,
                            "roi_id": rid,
                            "source_path": str(img_path),
                            "output_path": str(out_path),
                            "cx": int(cx),
                            "cy": int(cy),
                        }
                    )
            except Exception as e:
                print(f"ERROR processing {img_path}: {e}")
            finally:
                if use_pbar:
                    pbar.update(1)  # one tick per source image

    if manifest_rows:
        dfm = pd.DataFrame(manifest_rows)
        dfm.to_csv(output_root / "cropping_manifest.csv", index=False)
        print(
            f"Wrote manifest with {len(dfm)} rows: {output_root/'cropping_manifest.csv'}"
        )

    if use_pbar:
        pbar.close()
    print("Done.")


if __name__ == "__main__":
    main()


"""
example bash usage

python roi_cropping_pipeline.py \
    --ref-image ./data/raw/2025-10-02/PL_oc/Felix1_0.tiff \
    --images-root ./data/raw1/2025-10-02 \
    --modalities EL PL_oc PL_sc \
    --output-root ./data/processed/2025-10-02 \
    --crop-size 56 \
    --rows 4 --cols 4 \
    --preset chose4x4 \
    --hot-pixels true
"""
