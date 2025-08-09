#!/usr/bin/env python3
"""
DSEC → NBIN5 encoder.

Inputs (per-frame NPZ with x,y,t,p):
  /shared/qd8/event3d/DSEC/train/event_streams/<sequence>/<side>/*.npz

Outputs (one BigTIFF per NPZ, no side/N_BINS folders):
  /shared/qd8/event3d/DSEC/train/nbin_5_encoding/<sequence>/<same_name>.tif
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile
from tqdm import tqdm

# DSEC rectified resolution
H, W = 480, 640
NBIN = 5


def normalize(x: np.ndarray,
              relative_vmin: float = None,
              relative_vmax: float = None,
              interval_vmax: float = None) -> np.ndarray:
    vmax, vmin = float(x.max()), float(x.min())
    if relative_vmax is not None:
        vmax = vmin + relative_vmax
    if relative_vmin is not None:
        vmin = vmin + relative_vmin
    if interval_vmax is None:
        interval_vmax = max(vmax - vmin, 1e-9)
    x = x * (x >= vmin) * (x <= vmax)
    return (x - vmin) / interval_vmax
def nbin_encode(
    times: np.ndarray,
    polarity: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    height: int = H,   # 480 for DSEC
    width: int = W,    # 640 for DSEC
    nbin: int = NBIN   # 5 for DSEC
) -> np.ndarray:
    """
    STRICT port of your previous NBIN encoder.
    - Hard time binning via normalize(times)
    - Per-bin frame initialized to 128
    - Polarity mapping identical to old code
    - x,y cast to int (truncate), no rounding
    """
    # 1) integer pixel coords (truncate, like the old code)
    x = x.astype(np.int64)
    y = y.astype(np.int64)

    # 2) normalize time to [0,1]
    normalized_time = normalize(times)

    # 3) polarity to {0,255} with the exact same rules
    polarity_enc = polarity.copy()
    polarity_enc[polarity_enc == 1]  = 255
    polarity_enc[polarity_enc == -1] = 0
    polarity_enc[polarity_enc == 0]     = 0
    polarity_enc[polarity_enc == True]  = 255
    polarity_enc[polarity_enc == False] = 0
    polarity_enc = polarity_enc.astype(np.uint8, copy=False)

    # 4) output: (nbin, H, W) filled with 128
    encoded = np.ones((nbin, height, width), dtype=np.uint8) * 128

    # 5) assign events to discrete time bins
    time_bin = np.minimum((normalized_time * nbin).astype(int), nbin - 1)

    # 6) per-bin write (exact same structure as before)
    for b in range(nbin):
        frame = np.ones((height, width), dtype=np.uint8) * 128
        mask  = (time_bin == b)
        if np.any(mask):
            valid = (
                (y[mask] >= 0) & (y[mask] < height) &
                (x[mask] >= 0) & (x[mask] < width)
            )
            if np.any(valid):
                yv = y[mask][valid]
                xv = x[mask][valid]
                pv = polarity_enc[mask][valid]
                frame[yv, xv] = pv
        encoded[b] = frame

    return encoded


def load_npz_events(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(npz_path) as z:
        x = z["x"]  # float32
        y = z["y"]
        t = z["t"]  # int64 (µs)
        p = z["p"]  # uint8 {0,1}
    return x, y, t, p


def process_dsec_event_streams(events_root: str,
                               output_root: str,
                               side: str = "left",
                               nbin: int = NBIN,
                               overwrite: bool = False):
    events_root = Path(events_root)
    output_root = Path(output_root)
    assert events_root.is_dir(), f"events_root not found: {events_root}"
    output_root.mkdir(parents=True, exist_ok=True)

    sequences = sorted([d for d in events_root.iterdir() if d.is_dir()])
    total = 0

    for seq_dir in tqdm(sequences, desc="Sequences", unit="seq"):
        npz_dir = seq_dir / side
        if not npz_dir.is_dir():
            tqdm.write(f"[skip] {seq_dir.name}: no '{side}' dir")
            continue

        npz_files = sorted(npz_dir.glob("*.npz"))
        if not npz_files:
            tqdm.write(f"[skip] {seq_dir.name}: no NPZ files under {npz_dir}")
            continue

        out_dir = output_root / seq_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for npz_file in tqdm(npz_files, desc=f"  {seq_dir.name}", leave=False):
            out_tif = out_dir / (npz_file.stem + ".tif")
            if out_tif.exists() and not overwrite:
                continue

            try:
                x, y, t, p = load_npz_events(npz_file)
                if t.size == 0:
                    enc = np.full((nbin, H, W), 128, dtype=np.uint8)
                else:
                    enc = nbin_encode(t, p, x, y, height=H, width=W, nbin=nbin)

                assert enc.shape == (nbin, H, W)
                tifffile.imwrite(str(out_tif), enc, bigtiff=True)
                total += 1
            except Exception as e:
                tqdm.write(f"[err] {npz_file}: {e}")

    print(f"\n✅ Done: wrote {total} NBIN{nbin} files to {output_root}")


def main():
    ap = argparse.ArgumentParser("DSEC NBIN5 encoder (flat output tree)")
    ap.add_argument("--events-root", default="/shared/qd8/event3d/DSEC/train/event_streams",
                    help="Root with <sequence>/<side>/*.npz")
    ap.add_argument("--output-root", default="/shared/qd8/event3d/DSEC/train/nbin_5_encoding",
                    help="Where to write <sequence>/<same_name>.tif")
    ap.add_argument("--side", choices=["left", "right"], default="left",
                    help="Which side to read NPZs from (filename has no side component).")
    ap.add_argument("--nbin", type=int, default=NBIN)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    process_dsec_event_streams(args.events_root, args.output_root,
                               side=args.side, nbin=args.nbin, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
