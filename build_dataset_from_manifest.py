"""
Build per-cell .npz datasets from a JV manifest.

Inputs:
  - A manifest CSV with columns at least:
      date, channel, t_idx, t_hours, pce_fw, pce_rv,
      img_EL, img_PL_oc, img_PL_sc,
      has_EL, has_PL_oc, has_PL_sc, has_all_modalities

Outputs:
  dataset_root/
    meta.json
    cells/<cell_id>.npz

Each .npz contains:
  x:        (T, C, H, W) float32, channels: [EL_t, PLoc_t, PLsc_t, EL_0, PLoc_0, PLsc_0, rEL, rPLoc, rPLsc]
  t_idx:    (T,) int32
  pce_fw:   (T,) float32
  pce_rv:   (T,) float32
  pce_avg:  (T,) float32
  soh_fw:   (T,) float32   # vs reference image time
  soh_rv:   (T,) float32
  soh_avg:  (T,) float32
  ref_t_idx: () int32
  present_mask: (T, C) bool

CLI:
  python build_dataset_from_manifest.py \
    --manifest /path/to/processed/<date>/manifest.csv \
    --out-root /path/to/dataset_root \
    --average true \
    --include-deltas false \
    --eps-scale 1e-6
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import tifffile as tifi

    def imread(p):
        return tifi.imread(str(p))

except Exception:
    import imageio.v2 as imageio

    def imread(p):
        return imageio.imread(p)


def load_img_fp32(p: Path) -> np.ndarray:
    arr = imread(p)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] == 1 else np.mean(arr, axis=-1)
    return arr.astype(np.float32)


def build_stack_for_row(row, refs, eps_scale: float, include_deltas: bool):
    def _valid_path(v):
        return isinstance(v, str) and len(v) > 0 and Path(v).exists()

    paths = {
        "EL": row.get("img_EL", None),
        "PL_oc": row.get("img_PL_oc", None),
        "PL_sc": row.get("img_PL_sc", None),
    }
    if not all(_valid_path(v) for v in paths.values()):
        return None, False

    # read images at time t
    ELt = load_img_fp32(paths["EL"])
    PLoct = load_img_fp32(paths["PL_oc"])
    PLsct = load_img_fp32(paths["PL_sc"])
    # read reference images at time 0
    EL0 = refs["EL"]
    PLoc0 = refs["PL_oc"]
    PLsc0 = refs["PL_sc"]
    # compute per-modality ratios
    eps_EL = float(eps_scale * max(1.0, float(np.median(EL0))))
    eps_PLoc = float(eps_scale * max(1.0, float(np.median(PLoc0))))
    eps_PLsc = float(eps_scale * max(1.0, float(np.median(PLsc0))))
    rEL = ELt / (EL0 + eps_EL)
    rPLoc = PLoct / (PLoc0 + eps_PLoc)
    rPLsc = PLsct / (PLsc0 + eps_PLsc)
    # stack along channels
    chans = [ELt, PLoct, PLsct, EL0, PLoc0, PLsc0, rEL, rPLoc, rPLsc]
    # optional: compute per-modality deltas
    if include_deltas:
        dEL = ELt - EL0
        dPLoc = PLoct - PLoc0
        dPLsc = PLsct - PLsc0
        chans.extend([dEL, dPLoc, dPLsc])

    stack = np.stack(chans, axis=0).astype(np.float32)
    return stack, True


def pick_reference_row(df_cell: pd.DataFrame):
    def _valid_path(v):
        return isinstance(v, str) and len(v) > 0 and Path(v).exists()

    for _, r in df_cell.iterrows():
        if (
            _valid_path(r.get("img_EL"))
            and _valid_path(r.get("img_PL_oc"))
            and _valid_path(r.get("img_PL_sc"))
        ):
            return r

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument(
        "--average",
        type=str,
        default="true",
        help="If true, also compute PCE_avg=(FW+RV)/2 and SOH_avg vs image reference.",
    )
    ap.add_argument("--include-deltas", type=str, default="false")
    ap.add_argument("--eps-scale", type=float, default=1e-6)
    ap.add_argument("--min-timepoints", type=int, default=3)
    ap.add_argument(
        "--stack-map",
        type=str,
        default=None,
        help="Optional JSON/CSV mapping stack_id -> int code. If omitted, a map is auto-built from the manifest.",
    )

    args = ap.parse_args()

    use_avg = str(args.average).lower() in ("1", "true", "yes", "y")
    include_deltas = str(args.include_deltas).lower() in ("1", "true", "yes", "y")
    C = 9 + (3 if include_deltas else 0)

    man_path = Path(args.manifest)
    df = pd.read_csv(man_path)

    # NaN-safe for paths (already suggested earlier)
    for c in ("img_EL", "img_PL_oc", "img_PL_sc"):
        if c in df.columns:
            df[c] = df[c].where(df[c].notna(), "")

    # Ensure stack fields exist; if not, create simple fallbacks
    if "stack_id" not in df.columns:
        df["stack_id"] = ""
    if "sample_id" not in df.columns:
        df["sample_id"] = ""

    # Build or load stack_id -> int mapping
    stack_map = {}
    if args.stack_map:
        sp = Path(args.stack_map)
        if sp.suffix.lower() == ".json":
            stack_map = json.loads(sp.read_text(encoding="utf-8"))
        else:
            m = pd.read_csv(sp)
            stack_map = {str(r["stack_id"]): int(r["code"]) for _, r in m.iterrows()}

    def stack_code_of(sid: str) -> int:
        return stack_map.get(str(sid), -1)

    required_cols = [
        "date",
        "channel",
        "t_idx",
        "t_hours",
        "pce_fw",
        "pce_rv",
        "img_EL",
        "img_PL_oc",
        "img_PL_sc",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")

    out_root = Path(args.out_root) / df["date"].iloc[0]
    (out_root / "cells").mkdir(parents=True, exist_ok=True)

    groups = df.groupby("channel")
    from tqdm import tqdm as _tqdm

    pbar = _tqdm(total=len(groups), desc="Building cells", unit="cell")
    built = 0
    skipped = 0

    meta = {
        "channels": (
            [
                "EL_t",
                "PLoc_t",
                "PLsc_t",
                "EL_0",
                "PLoc_0",
                "PLsc_0",
                "rEL",
                "rPLoc",
                "rPLsc",
            ]
            + (["dEL", "dPLoc", "dPLsc"] if include_deltas else [])
        ),
        "stacks": list(stack_map.keys()),
        "stack_map": stack_map,
        "compute_average": use_avg,
        "crop_hw": None,
        "from_manifest": str(man_path),
        "eps_scale": args.eps_scale,
        "version": "0.1",
    }

    for cell_id, g in groups:
        g = g.sort_values("t_idx").reset_index(drop=True)
        # Per-cell identifiers (should be constant within group)
        cell_sample_id = str(g["sample_id"].iloc[0]) if "sample_id" in g.columns else ""
        cell_stack_id = str(g["stack_id"].iloc[0]) if "stack_id" in g.columns else ""
        cell_stack_code = stack_code_of(cell_stack_id)

        ref_row = pick_reference_row(g)
        if ref_row is None:
            skipped += 1
            pbar.update(1)
            continue
        try:
            EL0 = load_img_fp32(ref_row["img_EL"])
            PLoc0 = load_img_fp32(ref_row["img_PL_oc"])
            PLsc0 = load_img_fp32(ref_row["img_PL_sc"])
        except Exception as e:
            print(f"[{cell_id}] ERROR loading reference images: {e}")
            skipped += 1
            pbar.update(1)
            continue
        H, W = EL0.shape
        if meta["crop_hw"] is None:
            meta["crop_hw"] = [int(H), int(W)]
        refs = {"EL": EL0, "PL_oc": PLoc0, "PL_sc": PLsc0}

        stacks = []
        t_idx = []
        p_fw = []
        p_rv = []
        pmask = []
        for _, row in g.iterrows():
            stack, ok = build_stack_for_row(row, refs, args.eps_scale, include_deltas)
            if not ok:
                continue
            stacks.append(stack)
            t_idx.append(int(row["t_idx"]))
            p_fw.append(float(row["pce_fw"]) if pd.notnull(row["pce_fw"]) else np.nan)
            p_rv.append(float(row["pce_rv"]) if pd.notnull(row["pce_rv"]) else np.nan)
            pmask.append([True] * C)

        if len(stacks) < args.min_timepoints:
            skipped += 1
            pbar.update(1)
            continue

        X = np.stack(stacks, axis=0).astype(np.float32)
        TIDX = np.array(t_idx, dtype=np.int32)
        PFW = np.array(p_fw, dtype=np.float32)
        PRV = np.array(p_rv, dtype=np.float32)

        ref_p_fw = PFW[0] if np.isfinite(PFW[0]) else np.nan
        ref_p_rv = PRV[0] if np.isfinite(PRV[0]) else np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            SOH_FW = (
                PFW / ref_p_fw
                if np.isfinite(ref_p_fw) and ref_p_fw != 0
                else np.full_like(PFW, np.nan)
            )
            SOH_RV = (
                PRV / ref_p_rv
                if np.isfinite(ref_p_rv) and ref_p_rv != 0
                else np.full_like(PRV, np.nan)
            )

        if use_avg:
            PAVG = (PFW + PRV) / 2.0
            ref_p_avg = PAVG[0] if np.isfinite(PAVG[0]) else np.nan
            with np.errstate(divide="ignore", invalid="ignore"):
                SOH_AVG = (
                    PAVG / ref_p_avg
                    if np.isfinite(ref_p_avg) and ref_p_avg != 0
                    else np.full_like(PAVG, np.nan)
                )

        out_path = out_root / "cells" / f"{cell_id}.npz"
        np.savez_compressed(
            out_path,
            sample_id=cell_sample_id,
            stack_id=cell_stack_id,
            stack_code=np.int32(cell_stack_code),
            x=X,
            t_idx=TIDX,
            # add these two when use_avg:
            **({"pce_avg": PAVG, "soh_avg": SOH_AVG} if use_avg else {}),
            pce_fw=PFW,
            pce_rv=PRV,
            soh_fw=SOH_FW,
            soh_rv=SOH_RV,
            ref_t_idx=int(t_idx[0]),
            present_mask=np.array(pmask, dtype=bool),
        )
        built += 1
        pbar.set_postfix_str(f"last={cell_id}")
        pbar.update(1)

    pbar.close()
    with open(out_root / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Built {built} cells, skipped {skipped}. Saved to: {out_root}")


if __name__ == "__main__":
    main()
