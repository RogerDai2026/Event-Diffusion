#!/usr/bin/env python3
"""
MVSEC → Voxel (NBIN=5) encoder (paper-accurate).

Inputs (per-frame NPZ with keys x,y,t,p):
  /shared/qd8/event3d/MVSEC/outdoor_day2_events/<side>/*.npz

Outputs (one NPZ per frame):
  /shared/qd8/event3d/MVSEC/outdoor_day2_vox5/<same_name>.npz
    - key 'vox': float32 array of shape (NBIN, H, W)

Optional visualization (debug only):
  --save-vis writes an 8-bit TIFF next to each NPZ.
"""

import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
import tifffile
from tqdm import tqdm

# MVSEC DAVIS rectified size
H, W = 260, 346
NBIN = 3
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
    height: int,
    width: int,
    nbin: int,
) -> np.ndarray:
    """
    Legacy NBIN encoder:
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
    normalized_time = normalize(times)  # your helper

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


def normalize_nonzero_inplace(grid: np.ndarray) -> np.ndarray:
    """Match EventPreprocessor: only non-zeros get normalized to mean 0, std 1."""
    nz = (grid != 0)
    n = int(nz.sum())
    if n == 0:
        return grid
    vals = grid[nz].astype(np.float32, copy=False)
    mean = float(vals.mean())
    var  = float((vals * vals).mean() - mean * mean)
    std  = np.sqrt(max(var, 1e-12))
    grid[nz] = (vals - mean) / std
    return grid


# ---------- I/O helpers ----------

def load_npz_events(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(npz_path) as z:
        x = z["x"]            # int16
        y = z["y"]            # int16
        t = z["t"]            # float64 seconds
        p = z["p"]            # int8 in {-1,0,+1}
    return x, y, t, p


def save_vis_tiff(out_path_tif: Path, vox: np.ndarray):
    """Simple per-bin 8-bit visualization for sanity checking only."""
    # scale each bin independently to 0..255; keep zeros at 128 for background contrast
    vis = []
    for b in range(vox.shape[0]):
        v = vox[b]
        if np.all(v == 0):
            vis_bin = np.full(v.shape, 128, np.uint8)
        else:
            nz = (v != 0)
            vmin, vmax = float(v[nz].min()), float(v[nz].max())
            if vmax <= vmin:
                vis_bin = np.full(v.shape, 128, np.uint8)
            else:
                scaled = (v - vmin) / (vmax - vmin)  # 0..1 on non-zeros
                vis_bin = np.full(v.shape, 128, np.uint8)
                vis_bin[nz] = (scaled[nz] * 255.0).round().clip(0, 255).astype(np.uint8)
        vis.append(vis_bin)
    vis = np.stack(vis, axis=0)
    tifffile.imwrite(str(out_path_tif), vis, bigtiff=True)


# ---------- Main pipeline ----------
def process_mvsec_event_frames_oldnbin(events_root: str,
                                       output_root: str,
                                       side: str = "left",
                                       nbin: int = 3,
                                       overwrite: bool = False):
    """
    MVSEC → old NBIN encoder (uint8) with hard time bins and {0,128,255} polarity.
    Writes BigTIFFs shaped (nbin, H, W).
    """
    events_root = Path(events_root)
    output_root = Path(output_root)
    assert events_root.is_dir(), f"events_root not found: {events_root}"
    output_root.mkdir(parents=True, exist_ok=True)

    # Find directories that actually contain NPZs
    candidates = []
    if (events_root / side).is_dir():
        candidates.append(events_root / side)
    elif any(p.suffix == ".npz" for p in events_root.iterdir()):
        candidates.append(events_root)
    else:
        for d in sorted(p for p in events_root.iterdir() if p.is_dir()):
            if (d / side).is_dir():
                candidates.append(d / side)

    if not candidates:
        raise FileNotFoundError(f"No NPZ directory found under {events_root} (checked '{side}' and direct NPZs)")

    total = 0
    for npz_dir in candidates:
        npz_files = sorted(npz_dir.glob("*.npz"))
        if not npz_files:
            tqdm.write(f"[skip] no NPZ files under {npz_dir}")
            continue

        desc = f"{npz_dir.name}" if npz_dir != events_root else "mvsec"
        for npz_file in tqdm(npz_files, desc=f"  {desc}", leave=False):
            out_tif = output_root / (npz_file.stem + ".tif")
            if out_tif.exists() and not overwrite:
                continue

            try:
                x, y, t, p = load_npz_events(npz_file)
                if t.size == 0:
                    enc = np.full((nbin, H, W), 128, dtype=np.uint8)
                else:
                    # safety: enforce non-decreasing time
                    order = np.argsort(t, kind="stable")
                    if not np.all(order == np.arange(t.size)):
                        x, y, p, t = x[order], y[order], p[order], t[order]

                    # old NBIN (uint8) — NOTE: t is seconds (float64) in MVSEC
                    enc = nbin_encode(
                        times=t.astype(np.float64, copy=False),
                        polarity=p.astype(np.int8, copy=False),
                        x=x.astype(np.float32, copy=False),
                        y=y.astype(np.float32, copy=False),
                        height=H, width=W, nbin=nbin
                    )

                assert enc.shape == (nbin, H, W)
                tifffile.imwrite(str(out_tif), enc, bigtiff=True)
                total += 1

            except Exception as e:
                tqdm.write(f"[err] {npz_file}: {e}")

    print(f"\n✅ Done: wrote {total} NBIN{nbin} TIFFs to {output_root}")


def process_mvsec_event_frames(events_root: str,
                               output_root: str,
                               side: str = "left",
                               nbin: int = NBIN,
                               overwrite: bool = False,
                               normalize: bool = True,
                               save_vis: bool = False):
    events_root = Path(events_root)
    output_root = Path(output_root)
    assert events_root.is_dir(), f"events_root not found: {events_root}"
    output_root.mkdir(parents=True, exist_ok=True)

    # Find directories that actually contain NPZs
    candidates = []
    if (events_root / side).is_dir():
        candidates.append(events_root / side)
    elif any(p.suffix == ".npz" for p in events_root.iterdir()):
        candidates.append(events_root)
    else:
        for d in sorted(p for p in events_root.iterdir() if p.is_dir()):
            if (d / side).is_dir():
                candidates.append(d / side)

    if not candidates:
        raise FileNotFoundError(f"No NPZ directory found under {events_root} (checked '{side}' and direct NPZs)")

    total = 0
    for npz_dir in candidates:
        npz_files = sorted(npz_dir.glob("*.npz"))
        if not npz_files:
            tqdm.write(f"[skip] no NPZ files under {npz_dir}")
            continue

        desc = f"{npz_dir.name}" if npz_dir != events_root else "mvsec"
        for npz_file in tqdm(npz_files, desc=f"  {desc}", leave=False):
            out_npz = output_root / (npz_file.stem + ".npz")
            out_tif = output_root / (npz_file.stem + ".tif")

            if out_npz.exists() and not overwrite:
                continue

            try:
                x, y, t, p = load_npz_events(npz_file)
                if t.size == 0:
                    vox = np.zeros((nbin, H, W), np.float32)
                else:
                    # safety: ensure time is non-decreasing (our slicer should already guarantee this)
                    order = np.argsort(t, kind="stable")
                    if not np.all(order == np.arange(t.size)):
                        x, y, p, t = x[order], y[order], p[order], t[order]

                    # stack to [N,4]: [timestamp, x, y, polarity]
                    events = np.column_stack([
                        t.astype(np.float64, copy=False),
                        x.astype(np.int64,   copy=False),
                        y.astype(np.int64,   copy=False),
                        p.astype(np.float32, copy=False),
                    ])
                    # paper-accurate voxelization (triangular + accumulation)
                    vox = events_to_voxel_grid_numpy(events, nbin, W, H)

                    # match EventPreprocessor: normalize only non-zero entries
                    if normalize:
                        vox = normalize_nonzero_inplace(vox)

                # write float32 voxel grid
                np.savez_compressed(out_npz, vox=vox.astype(np.float32, copy=False))
                total += 1

                # optional visualization (debug)
                if save_vis:
                    save_vis_tiff(out_tif, vox)

            except Exception as e:
                tqdm.write(f"[err] {npz_file}: {e}")

    print(f"\n✅ Done: wrote {total} voxel files to {output_root}")


def main():
    ap = argparse.ArgumentParser("MVSEC voxel (NBIN=5) encoder — paper-accurate")
    ap.add_argument("--events-root", default="/shared/qd8/event3d/MVSEC/outdoor_night2_events",
                    help="Root with <side>/*.npz, or point directly to the <side> folder.")
    ap.add_argument("--output-root", default="/shared/qd8/event3d/MVSEC/outdoor_night2_old_nbin3",
                    help="Where to write <same_name>.npz with key 'vox'")
    ap.add_argument("--side", choices=["left", "right"], default="left")
    ap.add_argument("--nbin", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-normalize", action="store_true",
                    help="Skip non-zero mean/std normalization (default is to normalize).")
    ap.add_argument("--save-vis", action="store_true",
                    help="Also write an 8-bit TIFF per frame for quick visual checks.")
    ap.add_argument("--encoder", choices=["tri-accum", "old-nbin"], default="old-nbin",
                    help="tri-accum: paper-accurate float voxels (.npz); old-nbin: legacy uint8 per-bin frames (.tif)")
    args = ap.parse_args()

    if args.encoder == "tri-accum":
        process_mvsec_event_frames(
            events_root=args.events_root,
            output_root=args.output_root,
            side=args.side,
            nbin=args.nbin,
            overwrite=args.overwrite,
            normalize=(not args.no_normalize),
            # save_vis=args.save_vis
        )
    else:
        process_mvsec_event_frames_oldnbin(
            events_root=args.events_root,
            output_root=args.output_root,
            side=args.side,
            nbin=args.nbin,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
