#!/usr/bin/env python3
"""
DSEC → NBIN5 encoder with dual encoding options.

Supports three encoding methods:
  1. "mvsec": MVSEC-style hard binning
     - Time normalization to [0,1] with quantile-based clipping
     - Hard binning into discrete temporal slots  
     - Polarity: >0 → 255, ≤0 → 0
     - Background: 128 (mid-gray)
     - Output: uint8
     
  2. "dsec_legacy": Original DSEC method
     - Similar to MVSEC but with different polarity mapping rules
     - Hard binning with normalize function
     - Polarity mapping: 1→255, -1→0, 0→0, True→255, False→0
     - Background: 128 (mid-gray)
     - Output: uint8
     
  3. "voxelizer": Official E2VID/E2Depth voxelizer
     - Bilinear interpolation in time (triangular kernel)
     - Polarity preserved as signed values {-1, +1}
     - Background: 0
     - Output: float32

Inputs (per-frame NPZ with x,y,t,p):
  /shared/qd8/event3d/DSEC/train/event_streams/<sequence>/<side>/*.npz

Outputs (one BigTIFF per NPZ, no side/N_BINS folders):
  /shared/qd8/event3d/DSEC/train/nbin_5_encoding/<sequence>/<same_name>.tif

Usage:
  python generate_nbin_encoding_dsec.py --encoding-method mvsec
  python generate_nbin_encoding_dsec.py --encoding-method voxelizer --nbin 5
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


# ---------- MVSEC-style NBIN encoding (legacy method) ----------

def normalize_mvsec(x: np.ndarray,
                    relative_vmin: float = None,
                    relative_vmax: float = None,
                    interval_vmax: float = None) -> np.ndarray:
    """
    MVSEC-style normalization function.
    """
    vmax, vmin = float(x.max()), float(x.min())
    if relative_vmax is not None:
        vmax = vmin + relative_vmax
    if relative_vmin is not None:
        vmin = vmin + relative_vmin
    if interval_vmax is None:
        interval_vmax = max(vmax - vmin, 1e-9)
    x = x * (x >= vmin) * (x <= vmax)
    return (x - vmin) / interval_vmax

def nbin_encode_mvsec(
    times: np.ndarray,
    polarity: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    height: int,
    width: int,
    nbin: int,
) -> np.ndarray:
    """
    MVSEC-style NBIN encoder:
      - hard binning of time into nbin slices
      - per-bin frame initialized to 128 (mid-gray)
      - polarity > 0 → 255, polarity <= 0 → 0
      - last-write-wins at (y, x)
    Works with polarity in {-1,0,+1}, {0,1}, or bool.
    """
    # 1) integer pixel coords (truncate)
    x = x.astype(np.int64, copy=False)
    y = y.astype(np.int64, copy=False)

    # 2) normalize time to [0,1]
    normalized_time = normalize_mvsec(times)  # MVSEC helper

    # 3) uint8 polarity map via masks (no overflow)
    pos_mask = (polarity > 0) | (polarity == True)
    pv = np.zeros(polarity.shape, dtype=np.uint8)
    pv[pos_mask] = 255  # negatives & zeros stay 0

    # 4) output: (nbin, H, W) filled with 128
    encoded = np.full((nbin, height, width), 128, dtype=np.uint8)

    # 5) assign events to discrete time bins
    time_bin = np.minimum((normalized_time * nbin).astype(int), nbin - 1)

    # 6) per-bin write (last-write-wins)
    for b in range(nbin):
        frame = np.full((height, width), 128, dtype=np.uint8)
        mask  = (time_bin == b)
        if np.any(mask):
            xv = x[mask]; yv = y[mask]; pvb = pv[mask]
            valid = (yv >= 0) & (yv < height) & (xv >= 0) & (xv < width)
            if np.any(valid):
                frame[yv[valid], xv[valid]] = pvb[valid]
        encoded[b] = frame
    return encoded

# ---------- Official voxelizer (NumPy port, no changes in logic) ----------

def events_to_voxel_grid_numpy(events: np.ndarray, num_bins: int, width: int, height: int) -> np.ndarray:
    """
    Build a voxel grid with bilinear interpolation in time (triangular kernel),
    matching the official implementation.

    events: [N,4] -> [timestamp, x, y, polarity]
    returns: float32 voxel grid of shape (num_bins, height, width)
    """
    assert events.shape[1] == 4
    assert num_bins > 0 and width > 0 and height > 0

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize timestamps to [0, num_bins-1]
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp
    if deltaT == 0:
        deltaT = 1.0

    events = events.copy()
    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT

    ts = events[:, 0]
    xs = events[:, 1].astype(np.int64)
    ys = events[:, 2].astype(np.int64)
    pols = events[:, 3].astype(np.float32)
    pols[pols == 0] = -1.0  # map 0→-1 to get {−1,+1}

    tis = ts.astype(np.int64)              # left bin
    dts = ts - tis                         # fractional part
    vals_left  = pols * (1.0 - dts)
    vals_right = pols * dts

    # left bin accumulation
    valid = (tis >= 0) & (tis < num_bins) & (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    np.add.at(voxel_grid,
              xs[valid] + ys[valid] * width + tis[valid] * width * height,
              vals_left[valid])

    # right bin accumulation
    tis1 = tis + 1
    valid = (tis1 >= 0) & (tis1 < num_bins) & (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    np.add.at(voxel_grid,
              xs[valid] + ys[valid] * width + tis1[valid] * width * height,
              vals_right[valid])

    return voxel_grid.reshape(num_bins, height, width)



def normalize_dsec_legacy(x: np.ndarray,
                          relative_vmin: float = None,
                          relative_vmax: float = None,
                          interval_vmax: float = None) -> np.ndarray:
    """
    Original DSEC normalization (kept for backward compatibility).
    """
    vmax, vmin = float(x.max()), float(x.min())
    if relative_vmax is not None:
        vmax = vmin + relative_vmax
    if relative_vmin is not None:
        vmin = vmin + relative_vmin
    if interval_vmax is None:
        interval_vmax = max(vmax - vmin, 1e-9)
    x = x * (x >= vmin) * (x <= vmax)
    return (x - vmin) / interval_vmax

def nbin_encode_dsec_legacy(
    times: np.ndarray,
    polarity: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    height: int = H,   # 480 for DSEC
    width: int = W,    # 640 for DSEC
    nbin: int = NBIN   # 5 for DSEC
) -> np.ndarray:
    """
    Original DSEC NBIN encoder (kept for backward compatibility).
    - Hard time binning via normalize_dsec_legacy(times)
    - Per-bin frame initialized to 128
    - Polarity mapping identical to old code
    - x,y cast to int (truncate), no rounding
    """
    # 1) integer pixel coords (truncate, like the old code)
    x = x.astype(np.int64)
    y = y.astype(np.int64)

    # 2) normalize time to [0,1]
    normalized_time = normalize_dsec_legacy(times)

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
                               overwrite: bool = False,
                               encoding_method: str = "mvsec"):
    events_root = Path(events_root)
    output_root = Path(output_root)
    assert events_root.is_dir(), f"events_root not found: {events_root}"
    output_root.mkdir(parents=True, exist_ok=True)

    sequences = sorted([d for d in events_root.iterdir() if d.is_dir()])
    total = 0

    for seq_dir in tqdm(sequences, desc="Sequences", unit="seq"):
        npz_dir = seq_dir
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
                    if encoding_method == "voxelizer":
                        enc = np.zeros((nbin, H, W), dtype=np.float32)
                    else:  # mvsec or dsec_legacy
                        enc = np.full((nbin, H, W), 128, dtype=np.uint8)
                else:
                    if encoding_method == "mvsec":
                        enc = nbin_encode_mvsec(t, p, x, y, height=H, width=W, nbin=nbin)
                    elif encoding_method == "dsec_legacy":
                        enc = nbin_encode_dsec_legacy(t, p, x, y, height=H, width=W, nbin=nbin)
                    elif encoding_method == "voxelizer":
                        # Official voxelizer expects events as [N,4] array: [timestamp, x, y, polarity]
                        # Convert p from {0,1} to {-1,+1} if needed
                        p_signed = p.astype(np.float32)
                        p_signed[p_signed == 0] = -1.0
                        events = np.column_stack([t.astype(np.float32), x, y, p_signed])
                        enc = events_to_voxel_grid_numpy(events, nbin, W, H)
                    else:
                        raise ValueError(f"Unknown encoding_method: {encoding_method}")

                assert enc.shape == (nbin, H, W)
                tifffile.imwrite(str(out_tif), enc, bigtiff=True)
                total += 1
            except Exception as e:
                tqdm.write(f"[err] {npz_file}: {e}")

    print(f"\n✅ Done: wrote {total} NBIN{nbin} files to {output_root}")


def main():
    ap = argparse.ArgumentParser("DSEC NBIN5 encoder with dual encoding options")
    ap.add_argument("--events-root", default="/shared/qd8/event3d/DSEC/train/event_streams",
                    help="Root with <sequence>/<side>/*.npz")
    ap.add_argument("--output-root", default="/shared/qd8/event3d/DSEC/train/nbin_3_encoding",
                    help="Where to write <sequence>/<same_name>.tif")
    ap.add_argument("--side", choices=["left", "right"], default="left",
                    help="Which side to read NPZs from (filename has no side component).")
    ap.add_argument("--nbin", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--encoding-method", 
                    choices=["mvsec", "dsec_legacy", "voxelizer"], 
                    default="dsec_legacy",
                    help="Encoding method: 'mvsec' (MVSEC-style hard binning, uint8), "
                         "'dsec_legacy' (original DSEC method, uint8), "
                         "'voxelizer' (official bilinear interpolation, float32)")
    args = ap.parse_args()

    print(f"Using encoding method: {args.encoding_method}")
    if args.encoding_method == "mvsec":
        print("  - MVSEC-style hard binning with 128 background, uint8 output")
    elif args.encoding_method == "dsec_legacy":
        print("  - Original DSEC method with 128 background, uint8 output")
    elif args.encoding_method == "voxelizer":
        print("  - Official voxelizer with bilinear interpolation, float32 output")

    process_dsec_event_streams(args.events_root, args.output_root,
                               side=args.side, nbin=args.nbin, overwrite=args.overwrite,
                               encoding_method=args.encoding_method)


if __name__ == "__main__":
    main()
