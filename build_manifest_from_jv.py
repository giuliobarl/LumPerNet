"""
Build a manifest aligning JV rows with available EL/PL images by channel & time index.

Given a directory layout:
data/
  raw/<date>/JV/<channel>/<...txt>
  processed/<date>/<modality>/<channel>/<index>.tif

This script:
- Parses each channel JV .txt by finding "## Data ##" then taking the next line as the header.
- Creates one row per JV row (t_idx), adding paths for EL, PL_oc, PL_sc images if present.
- Writes a CSV manifest at: data/processed/<date>/jv_manifest.csv

Usage:
  python build_manifest_from_jv.py \
      --data-root /path/to/data \
      --date 2025-10-02 \
      --modalities EL PL_oc PL_sc

Notes:
- Image lookup is index-based: it looks for files named "<t_idx>.tif" or "<t_idx>.tiff".
- If your files use zero-padded indices, adapt the parser or add a mapping step later.
"""

import argparse
import re
import sys
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def find_data_marker(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        if "## Data" in line:
            return i
    return None


def parse_sample_and_channel_from_jv(jv_path: Path) -> tuple[str, str, str]:
    """
    JV filenames look like: ..._Stability (Parameters)_{STACK}_{BATCH}-{CHANNEL}.txt
    Example: 0000_2025-10-13_11.55.43_Stability (Parameters)_Felix_new-1A.txt
    Returns (stack_id, sample_id, channel_from_name). If pattern fails, returns ("", "", "").
    """
    stem = jv_path.stem
    if "_Stability (Parameters)_" in stem:
        right = stem.split("_Stability (Parameters)_", 1)[1]
        if "-" in right:
            sample, ch = right.rsplit("-", 1)
            if "_" in sample:
                _, stack = sample.rsplit("_", 1)
            return stack, sample, ch
    return "", "", ""


def parse_jv_file(jv_path: Path) -> pd.DataFrame:
    # Find the "## Data ##" line number
    lines = jv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    data_idx = next(i for i, ln in enumerate(lines) if "## Data" in ln)

    # Skip everything up to (and including) the marker line;
    # next line is the header row
    df = pd.read_csv(
        jv_path,
        engine="python",
        sep="\t",  # <-- force TSV
        skiprows=range(0, data_idx + 1),
        header=0,
        index_col=False,
    )

    # Normalize headers
    df.columns = [re.sub(r"\s+", " ", c.strip().lower()) for c in df.columns]

    # Positional index as t_idx
    df = df.reset_index(drop=True)
    df["t_idx"] = df.index.astype(int)

    # Robust column selection (works across slight header variations)
    time_col = next((c for c in df.columns if "time" in c and "hour" in c), None)
    eff_fw = next((c for c in df.columns if "efficiency" in c and "fw" in c), None)
    eff_rv = next((c for c in df.columns if "efficiency" in c and "rv" in c), None)
    pmpp_fw = next((c for c in df.columns if "p_mpp" in c and "fw" in c), None)
    pmpp_rv = next((c for c in df.columns if "p_mpp" in c and "rv" in c), None)
    voc_fw = next((c for c in df.columns if "voc" in c and "fw" in c), None)
    voc_rv = next((c for c in df.columns if "voc" in c and "rv" in c), None)
    jsc_fw = next((c for c in df.columns if "jsc" in c and "fw" in c), None)
    jsc_rv = next((c for c in df.columns if "jsc" in c and "rv" in c), None)
    ff_fw = next((c for c in df.columns if "fill" in c and "fw" in c), None)
    ff_rv = next((c for c in df.columns if "fill" in c and "rv" in c), None)

    assert (
        time_col and (eff_fw or pmpp_fw) and (eff_rv or pmpp_rv)
    ), "Missing expected JV columns"
    if eff_fw and pmpp_fw:
        ratio = np.nanmedian(
            pd.to_numeric(df[eff_fw], errors="coerce")
            / pd.to_numeric(df[pmpp_fw], errors="coerce")
        )
        if not (0.8 <= ratio <= 1.2):
            print(
                f"WARNING: Efficiency and P_MPP differ (median ratio ~{ratio:.2f}) in {jv_path.name}"
            )

    out = pd.DataFrame(
        {
            "t_idx": df["t_idx"],
            "t_hours": (
                pd.to_numeric(df[time_col], errors="coerce") if time_col else pd.NA
            ),
            "pce_fw": (
                pd.to_numeric(df[eff_fw], errors="coerce")
                if eff_fw
                else (pd.to_numeric(df[pmpp_fw], errors="coerce") if pmpp_fw else pd.NA)
            ),
            "pce_rv": (
                pd.to_numeric(df[eff_rv], errors="coerce")
                if eff_rv
                else (pd.to_numeric(df[pmpp_rv], errors="coerce") if pmpp_rv else pd.NA)
            ),
            "voc_fw": df[voc_fw],
            "voc_rv": df[voc_rv],
            "jsc_fw": df[jsc_fw],
            "jsc_rv": df[jsc_rv],
            "ff_fw": df[ff_fw],
            "ff_rv": df[ff_rv],
        }
    )

    return out


def scan_channel_images(
    processed_date_dir: Path, channel: str, modalities: list[str]
) -> dict[str, dict[int, Path]]:
    """
    Return { modality: { t_idx: path } } for available images named '<t_idx>.tif' or '<t_idx>.tiff'.
    """
    out: dict[str, dict[int, Path]] = {}
    for m in modalities:
        base = processed_date_dir / m
        cands = [
            d
            for d in base.iterdir()
            if d.is_dir()
            and (
                d.name.endswith(f"_{channel}")
                or d.name.endswith(f"-{channel}")
                or d.name == channel
            )
        ]
        if not cands:
            continue
        mod_dir = cands[0]
        mod_map: dict[int, Path] = {}
        if mod_dir.exists():
            for p in mod_dir.iterdir():
                if not p.is_file():
                    continue
                if p.suffix.lower() not in (".tif", ".tiff"):
                    continue
                stem = p.stem
                # Accept pure integer stems; ignore others
                try:
                    t_idx = int(stem)
                except ValueError:
                    # Try to catch names like 't_0068'
                    mobj = re.match(r"t_(\d+)$", stem)
                    if mobj:
                        t_idx = int(mobj.group(1))
                    else:
                        continue
                # Prefer .tif over .tiff if both exist; last one wins
                mod_map[t_idx] = p
        out[m] = mod_map
    return out


def build_manifest_for_channel(
    data_root: Path, date: str, channel: str, modalities: list[str], jv_file: Path
) -> pd.DataFrame:
    # raw_date_dir = data_root / "raw" / date
    proc_date_dir = data_root / "processed" / date

    # Parse JV
    df_jv = parse_jv_file(jv_file)
    df_jv["channel"] = channel
    df_jv["jv_file"] = str(jv_file)

    stack_id, sample_id, ch_from_name = parse_sample_and_channel_from_jv(jv_file)
    # prefer directory 'channel' for the electrical channel, but keep what's in the filename for sanity checks
    df_jv["sample_id"] = sample_id
    df_jv["stack_id"] = stack_id

    # Scan images
    img_maps = scan_channel_images(proc_date_dir, channel, modalities)

    # Build manifest rows: one row per JV index
    rows = []
    for t_idx, row in df_jv.reset_index(drop=True).iterrows():
        t_idx = int(t_idx)
        entry = {
            "date": date,
            "sample_id": row.get("sample_id", ""),
            "stack_id": row.get("stack_id", ""),
            "channel": channel,
            "t_idx": t_idx,
            "t_hours": row.get("t_hours", np.nan),
            "pce_fw": row.get("pce_fw", np.nan),
            "pce_rv": row.get("pce_rv", np.nan),
            "voc_fw": row.get("voc_fw", np.nan),
            "voc_rv": row.get("voc_rv", np.nan),
            "jsc_fw": row.get("jsc_fw", np.nan),
            "jsc_rv": row.get("jsc_rv", np.nan),
            "ff_fw": row.get("ff_fw", np.nan),
            "ff_rv": row.get("ff_rv", np.nan),
            "jv_file": str(jv_file),
        }
        has_all = True
        for m in modalities:
            p = img_maps.get(m, {}).get(t_idx, None)
            entry[f"img_{m}"] = str(p) if p is not None else ""
            entry[f"has_{m}"] = bool(p is not None)
            if p is None:
                has_all = False
        entry["has_all_modalities"] = has_all
        rows.append(entry)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Build manifest aligning JV rows to available images for each channel."
    )
    ap.add_argument(
        "--data-root", type=str, required=True, help="Path to the 'data' root folder."
    )
    ap.add_argument(
        "--date",
        type=str,
        required=True,
        help="Experiment date subfolder under raw/ and processed/.",
    )
    ap.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=["EL", "PL_oc", "PL_sc"],
        help="Modalities to index (folders under processed/date).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path for the manifest CSV. Default: data/processed/<date>/jv_manifest.csv",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root)
    date = args.date
    modalities = args.modalities
    proc_out = (
        Path(args.out)
        if args.out
        else (data_root / "processed" / date / "jv_manifest.csv")
    )
    proc_out.parent.mkdir(parents=True, exist_ok=True)

    # JV root: data/raw/<date>/JV
    jv_root = data_root / "raw" / date / "JV"
    if not jv_root.exists():
        print(f"ERROR: JV folder not found: {jv_root}")
        sys.exit(1)

    # Iterate channels: subfolders under JV
    channel_dirs = [d for d in jv_root.iterdir() if d.is_dir()]
    if not channel_dirs:
        print(f"WARNING: no channel subfolders found under {jv_root}")

    all_rows = []
    for ch_dir in channel_dirs:
        channel = ch_dir.name
        # Expect exactly one JV file per channel; if multiple, pick the latest by mtime
        jv_files = sorted(
            ch_dir.rglob("*_Stability (Parameters)_*.txt"),
            key=lambda p: p.stat().st_mtime,
        )
        if not jv_files:
            print(f"WARNING: no JV .txt in {ch_dir}")
            continue
        jv_path = jv_files[-1]
        try:
            df_ch = build_manifest_for_channel(
                data_root, date, channel, modalities, jv_path
            )
            all_rows.append(df_ch)
            print(f"Processed {channel}: {len(df_ch)} JV rows")
        except Exception as e:
            print(f"ERROR parsing {jv_path}: {e}")

    if all_rows:
        df_all = pd.concat(all_rows, ignore_index=True)
        df_all.to_csv(proc_out, index=False)
        print(f"Wrote manifest: {proc_out} ({len(df_all)} rows)")
    else:
        print("No rows to write.")


if __name__ == "__main__":
    main()

"""
example bash usage

python build_manifest_from_jv.py \
  --data-root ./data \
  --date 2025-10-02 \
  --modalities EL PL_oc PL_sc
"""
