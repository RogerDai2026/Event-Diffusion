#!/usr/bin/env python3
"""
DSEC → per-depth-frame events and depth (separate NPZs).

Extracts events from DSEC H5 files and aligns them with depth timestamps.

Outputs (aligned by filename index k):
  events_out/0000000000.npz : x[float32], y[float32], t[int64], p[uint8∈{0,1}]
  depth_out/0000000000.npz  : depth[native dtype], (H,W) = (480,640)

Default windowing: centered 50 ms around each depth timestamp.
"""

import argparse
from pathlib import Path
import numpy as np
import h5py
import hdf5plugin  # Required for Blosc compression in DSEC files
from tqdm import tqdm
import os

# DSEC rectified resolution
H, W = 480, 640


def read_dsec_timestamps(timestamps_file: Path) -> np.ndarray:
    """
    Read DSEC depth timestamps from timestamps.txt file.
    Returns timestamps in microseconds as int64.
    """
    with open(timestamps_file, 'r') as f:
        timestamps = [int(line.strip()) for line in f if line.strip()]
    return np.array(timestamps, dtype=np.int64)


def open_dsec_events_reader(h5_path: Path):
    """
    Create a reader for DSEC H5 events.
    
    DSEC H5 structure:
    /events/x : (N,) uint16 - x coordinates
    /events/y : (N,) uint16 - y coordinates  
    /events/t : (N,) int64 - timestamps in microseconds
    /events/p : (N,) uint8 - polarity {0,1}
    /t_offset : scalar - time offset in microseconds to add to event timestamps
    
    Returns:
        get_slice(i0, i1) -> (x, y, t, p)
        t_vec: 1D int64 array of event timestamps (microseconds)
    """
    f = h5py.File(h5_path, 'r')
    
    events = f['events']
    x_ds = events['x']
    y_ds = events['y'] 
    t_ds = events['t']
    p_ds = events['p']
    
    # Read time offset
    t_offset = int(f['t_offset'][()])
    print(f"Time offset: {t_offset} microseconds")
    
    # Read timestamps for indexing (this might be large but needed for searchsorted)
    print(f"Loading timestamps from {h5_path}...")
    t_vec = np.array(t_ds[:], dtype=np.int64)
    # Apply time offset to align with image timestamps
    t_vec = t_vec + t_offset
    print(f"Applied time offset: {t_offset} μs")
    
    def get_slice(i0, i1):
        x = np.array(x_ds[i0:i1], dtype=np.float32)  # Convert to float32
        y = np.array(y_ds[i0:i1], dtype=np.float32)
        t = np.array(t_ds[i0:i1], dtype=np.int64)
        p = np.array(p_ds[i0:i1], dtype=np.uint8)
        # Apply time offset to align with image timestamps
        t = t + t_offset
        return x, y, t, p
    
    return get_slice, t_vec, f  # Return file handle to keep it open


def process_dsec_sequence(events_h5_path: Path,
                         depth_timestamps_path: Path,
                         events_out: Path,
                         depth_out: Path = None,
                         window_ms: float = 50.0,
                         start: int = 0,
                         limit: int = 0,
                         overwrite: bool = False):
    """
    Process a single DSEC sequence, extracting events aligned with depth timestamps.
    """
    events_out.mkdir(parents=True, exist_ok=True)
    if depth_out:
        depth_out.mkdir(parents=True, exist_ok=True)
    
    # Read depth timestamps
    depth_ts = read_dsec_timestamps(depth_timestamps_path)
    N = len(depth_ts)
    print(f"[info] depth frames: {N}")
    
    # Convert timestamps from microseconds to seconds for windowing
    depth_ts_sec = depth_ts.astype(np.float64) * 1e-6
    
    if N > 2:
        med_dt = float(np.median(np.diff(depth_ts_sec)))
        print(f"[info] median Δt between depth frames: {med_dt*1000:.2f} ms")
    
    # Open events reader
    get_slice, t_vec, h5_file = open_dsec_events_reader(events_h5_path)
    t_vec_sec = t_vec.astype(np.float64) * 1e-6  # Convert to seconds
    print(f"[info] events: N={t_vec.size}, t∈[{t_vec_sec[0]:.6f},{t_vec_sec[-1]:.6f}] s")
    
    # Build event windows per depth frame (centered windowing)
    half_window = 0.5 * (window_ms * 1e-3)
    lefts = np.searchsorted(t_vec_sec, depth_ts_sec - half_window, side="left")
    rights = np.searchsorted(t_vec_sec, depth_ts_sec + half_window, side="left")
    
    start_idx = max(start, 0)
    end_idx = N if limit == 0 else min(start_idx + limit, N)
    
    indices = [(k, int(lefts[k]), int(rights[k])) for k in range(start_idx, end_idx)]
    
    # Export loop
    saved = 0
    for k, a, b in tqdm(indices, desc="frames", unit="frm"):
        stem = f"{k:010d}"
        ev_path = events_out / f"{stem}.npz"
        
        # Events for the window [a,b)
        if overwrite or (not ev_path.exists()):
            if b > a:
                x, y, t, p = get_slice(a, b)
                
                # Bounds check
                if x.size and (x.max() >= W or y.max() >= H or x.min() < 0 or y.min() < 0):
                    print(f"Warning: Event coords out of bounds at frame {k}: "
                          f"x∈[{x.min():.1f},{x.max():.1f}], y∈[{y.min():.1f},{y.max():.1f}] (W={W}, H={H})")
                    # Clip coordinates to valid range
                    x = np.clip(x, 0, W-1)
                    y = np.clip(y, 0, H-1)
            else:
                # Empty window: write empty arrays to keep alignment
                x = np.empty((0,), dtype=np.float32)
                y = np.empty((0,), dtype=np.float32)
                t = np.empty((0,), dtype=np.int64)
                p = np.empty((0,), dtype=np.uint8)
            
            np.savez_compressed(
                ev_path,
                x=x.astype(np.float32, copy=False),
                y=y.astype(np.float32, copy=False),
                t=t.astype(np.int64, copy=False),
                p=p.astype(np.uint8, copy=False),
            )
            saved += 1
    
    h5_file.close()
    print(f"[done] wrote {saved} event files to {events_out}")


def main():
    ap = argparse.ArgumentParser("Extract DSEC events (x,y,t,p) aligned per depth frame")
    ap.add_argument("--events-h5", required=True,
                    help="Path to DSEC events H5 file (e.g., .../events/left/events.h5)")
    ap.add_argument("--depth-timestamps", required=True,
                    help="Path to depth timestamps.txt file")
    ap.add_argument("--events-out", required=True,
                    help="Output directory for event NPZ files")
    ap.add_argument("--depth-out", default=None,
                    help="Output directory for depth NPZ files (optional)")
    ap.add_argument("--window-ms", type=float, default=50.0,
                    help="Window length around each depth timestamp (ms)")
    ap.add_argument("--start", type=int, default=0,
                    help="First frame index to export (inclusive)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Export at most N frames (0=all)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    
    events_h5_path = Path(args.events_h5)
    depth_timestamps_path = Path(args.depth_timestamps)
    events_out = Path(args.events_out)
    depth_out = Path(args.depth_out) if args.depth_out else None
    
    if not events_h5_path.exists():
        raise FileNotFoundError(f"Events H5 file not found: {events_h5_path}")
    if not depth_timestamps_path.exists():
        raise FileNotFoundError(f"Depth timestamps file not found: {depth_timestamps_path}")
    
    process_dsec_sequence(
        events_h5_path=events_h5_path,
        depth_timestamps_path=depth_timestamps_path,
        events_out=events_out,
        depth_out=depth_out,
        window_ms=args.window_ms,
        start=args.start,
        limit=args.limit,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
