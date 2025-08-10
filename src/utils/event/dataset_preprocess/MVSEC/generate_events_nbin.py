#!/usr/bin/env python3
"""
MVSEC Outdoor Day2 → per-depth-frame events and depth (separate NPZs).

Outputs (aligned by filename index k):
  events_out/0000000000.npz : x[int16], y[int16], t[float64], p[int8∈{-1,0,+1}]
  depth_out/0000000000.npz  : depth[native dtype], (H,W) = (260,346)

Default windowing matches MVSEC preview: centered 50 ms around each depth timestamp.
"""

import argparse
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm


# ---------- Event readers ----------

def open_events_reader(ev_node, flat_order="x y t p"):
    """
    Create a lightweight reader over the events stream.

    Supports three layouts under /davis/<side>/events:
      1) Group with datasets: x, y, ts(or t), polarity/p/pol
      2) Compound dataset with named fields
      3) Flat numeric 2D array with 4 cols/rows (N,4) or (4,N), order via flat_order

    Returns:
      get_slice(i0, i1) -> (x, y, t, p)
      t_vec: 1D float64 array of event timestamps (seconds), length = N events
    """
    if isinstance(ev_node, h5py.Group):
        x = ev_node["x"]; y = ev_node["y"]
        t = ev_node["ts"] if "ts" in ev_node else ev_node["t"]
        p = ev_node.get("polarity") or ev_node.get("p") or ev_node.get("pol")
        if p is None:
            raise KeyError("events group missing polarity dataset (polarity/p/pol)")
        t_vec = np.asarray(t[:], dtype=np.float64)
        def get_slice(i0, i1):
            return (np.asarray(x[i0:i1]),
                    np.asarray(y[i0:i1]),
                    np.asarray(t[i0:i1], dtype=np.float64),
                    np.asarray(p[i0:i1]))
        return get_slice, t_vec

    if isinstance(ev_node, h5py.Dataset):
        ds = ev_node
        # Case 2: compound
        if ds.dtype.names:
            names = {n.lower(): n for n in ds.dtype.names}
            def pick(keys):
                for k in keys:
                    if k in names:
                        return names[k]
                raise KeyError(f"Missing field {keys} in compound events dtype.")
            kx = pick(["x"]); ky = pick(["y"])
            kt = pick(["ts", "t"]); kp = pick(["polarity", "p", "pol"])
            t_vec = np.asarray(ds[kt][:], dtype=np.float64)
            def get_slice(i0, i1):
                sl = ds[i0:i1]
                return (np.asarray(sl[kx]),
                        np.asarray(sl[ky]),
                        np.asarray(sl[kt], dtype=np.float64),
                        np.asarray(sl[kp]))
            return get_slice, t_vec

        # Case 3: flat numeric (N,4) or (4,N)
        cols = flat_order.strip().lower().split()
        if set(cols) != {"x", "y", "t", "p"} or len(cols) != 4:
            raise ValueError("--flat-order must be a permutation of 'x y t p'")
        if ds.ndim != 2 or 4 not in ds.shape:
            raise TypeError(f"Unsupported events dataset shape {ds.shape} at {ds.name}")

        idx = {name: cols.index(name) for name in ["x", "y", "t", "p"]}

        # Normalize row access to (N,4) views
        if ds.shape[0] == 4 and ds.shape[1] != 4:
            # (4,N) -> (N,4) view without copying
            def row_block(i0, i1): return ds[:, i0:i1].T
            t_vec = np.asarray(ds[idx["t"], :], dtype=np.float64)
        else:
            def row_block(i0, i1): return ds[i0:i1, :]
            t_vec = np.asarray(ds[:, idx["t"]], dtype=np.float64)

        def get_slice(i0, i1):
            sl = row_block(i0, i1)
            return (np.asarray(sl[:, idx["x"]]),
                    np.asarray(sl[:, idx["y"]]),
                    np.asarray(sl[:, idx["t"]], dtype=np.float64),
                    np.asarray(sl[:, idx["p"]]))
        return get_slice, t_vec

    raise TypeError(f"Unknown events node type: {type(ev_node)} at {getattr(ev_node, 'name', '')}")


def to_int8_pol(p, scheme="zpm1"):
    """
    Normalize polarity to int8.

    scheme:
      - "zpm1": clamp to {-1,0,+1} (default; preserves zeros if present)
      - "pm1" : force {-1,+1}; maps booleans/01 via 2p-1; zeros become -1
      - "01"  : keep {0,1} as-is (not used here)
    """
    if p.dtype == np.bool_:
        p = p.astype(np.int8)
        return (2 * p - 1).astype(np.int8) if scheme == "pm1" else p
    p = p.astype(np.int8, copy=False)
    if scheme == "pm1":
        if np.min(p) >= 0:  # {0,1}
            return (2 * p - 1).astype(np.int8)
        return np.where(p == 0, -1, np.sign(p)).astype(np.int8)
    if scheme == "01":
        return np.clip(p, 0, 1).astype(np.int8)
    return np.clip(np.sign(p), -1, 1).astype(np.int8)  # "zpm1"


# ---------- Depth readers ----------

def read_depth_arrays(dgrp):
    """
    Returns:
      depth_ts: (N,) float64 seconds
      depth_imgs: HDF5 dataset or ndarray, shape (N,H,W)
    Handles either a dataset or a group with 'data' and 'timestamps'.
    """
    if "depth_image_rect_ts" in dgrp:
        depth_ts = np.asarray(dgrp["depth_image_rect_ts"][:], dtype=np.float64)
    else:
        depth_ts = np.asarray(dgrp["depth_image_rect"]["timestamps"][:], dtype=np.float64)

    if "depth_image_rect" in dgrp and isinstance(dgrp["depth_image_rect"], h5py.Group):
        depth_imgs = dgrp["depth_image_rect"]["data"]  # (N,H,W)
    elif "depth_image_rect" in dgrp:
        depth_imgs = dgrp["depth_image_rect"]          # (N,H,W)
    else:
        raise KeyError("depth_image_rect not found under depth HDF5")

    return depth_ts, depth_imgs


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser("Export MVSEC events (x,y,t,p) and depth (separate NPZs), aligned per frame.")
    ap.add_argument("--data-h5", default="/shared/qd8/event3d/MVSEC/hdf5/outdoor_day/outdoor_day1_data.hdf5")
    ap.add_argument("--gt-h5",   default="/shared/qd8/event3d/MVSEC/hdf5/outdoor_day/outdoor_day1_gt.hdf5")
    ap.add_argument("--side",    choices=["left", "right"], default="left",
                    help="DAVIS side to use. MVSEC canonical frame is LEFT; default is left.")
    ap.add_argument("--events-out", default="/shared/qd8/event3d/MVSEC/outdoor_day1_events/left")
    ap.add_argument("--depth-out",  default="/shared/qd8/event3d/MVSEC/outdoor_day1_depth/left")
    ap.add_argument("--mode",    choices=["centered", "between"], default="centered",
                    help="centered: [t_k-Δ/2, t_k+Δ/2); between: (t_{k-1}, t_k]")
    ap.add_argument("--window-ms", type=float, default=50.0, help="Window length for centered mode (ms)")
    ap.add_argument("--start",   type=int, default=0, help="First frame index to export (inclusive)")
    ap.add_argument("--limit",   type=int, default=0, help="Export at most N frames (0=all)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--flat-order", default="x y t p",
                    help="Column order for flat events arrays (permute of 'x y t p')")
    ap.add_argument("--pol-scheme", choices=["zpm1", "pm1", "01"], default="zpm1",
                    help="How to normalize polarity before saving")
    args = ap.parse_args()

    ev_dir = Path(args.events_out); ev_dir.mkdir(parents=True, exist_ok=True)
    dp_dir = Path(args.depth_out);  dp_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.data_h5, "r") as fd, h5py.File(args.gt_h5, "r") as fg:
        # --- depth stream (LEFT by default) ---
        dgrp = fg["davis"][args.side]
        depth_ts, depth_imgs = read_depth_arrays(dgrp)
        N = len(depth_ts)
        print(f"[info] depth frames ({args.side}): {N}")

        # Assert MVSEC DAVIS size 260x346 (H,W)
        H, W = depth_imgs.shape[-2], depth_imgs.shape[-1]
        assert (H, W) == (260, 346), f"Expected DAVIS rect size 260x346, got {H}x{W}"

        # quick cadence check
        if N > 2:
            med_dt = float(np.median(np.diff(depth_ts)))
            print(f"[info] median Δt between depth frames: {med_dt*1000:.2f} ms")

        # --- events stream ---
        ev_node = fd["davis"][args.side]["events"]
        get_slice, t_vec = open_events_reader(ev_node, flat_order=args.flat_order)
        print(f"[info] events: N={t_vec.size}, t∈[{t_vec[0]:.6f},{t_vec[-1]:.6f}] s")

        # --- build event windows per depth frame ---
        indices = []
        if args.mode == "between":
            edges = np.searchsorted(t_vec, depth_ts, side="right")
            start_idx = max(args.start, 1)  # window needs previous frame
            end_idx = N if args.limit == 0 else min(start_idx + args.limit, N)
            for k in range(start_idx, end_idx):
                a, b = int(edges[k-1]), int(edges[k])
                indices.append((k, a, b))
        else:
            half = 0.5 * (args.window_ms * 1e-3)
            lefts  = np.searchsorted(t_vec, depth_ts - half, side="left")
            rights = np.searchsorted(t_vec, depth_ts + half, side="left")
            start_idx = max(args.start, 0)
            end_idx = N if args.limit == 0 else min(start_idx + args.limit, N)
            for k in range(start_idx, end_idx):
                a, b = int(lefts[k]), int(rights[k])
                indices.append((k, a, b))

        # --- export loop ---
        saved = 0
        for k, a, b in tqdm(indices, desc="frames", unit="frm"):
            stem = f"{k:010d}"
            ev_path = ev_dir / f"{stem}.npz"
            dp_path = dp_dir / f"{stem}.npz"

            # Depth (always write unless exists and not overwrite)
            if args.overwrite or (not dp_path.exists()):
                depth = np.asarray(depth_imgs[k])  # keep original dtype/units
                if depth.shape != (H, W):
                    raise ValueError(f"Depth[{k}] has shape {depth.shape}, expected {(H, W)}")
                np.savez_compressed(dp_path, depth=depth)

            # Events for the window [a,b)
            if args.overwrite or (not ev_path.exists()):
                if b > a:
                    x, y, t, p = get_slice(a, b)
                    # bounds check (fail fast if calibration mismatch)
                    if x.size and (x.max() >= W or y.max() >= H or x.min() < 0 or y.min() < 0):
                        raise ValueError(
                            f"Event coords out of bounds at frame {k}: "
                            f"x∈[{x.min()},{x.max()}], y∈[{y.min()},{y.max()}] (W={W}, H={H})"
                        )
                    p = to_int8_pol(p, scheme=args.pol_scheme)
                else:
                    # empty window: write empty arrays to keep alignment
                    x = np.empty((0,), dtype=np.int16)
                    y = np.empty((0,), dtype=np.int16)
                    t = np.empty((0,), dtype=np.float64)
                    p = np.empty((0,), dtype=np.int8)

                np.savez_compressed(
                    ev_path,
                    x=x.astype(np.int16, copy=False),
                    y=y.astype(np.int16, copy=False),
                    t=t.astype(np.float64, copy=False),
                    p=p.astype(np.int8, copy=False),
                )

            saved += 1

    print(f"[done] wrote {saved} aligned files:\n  events → {ev_dir}\n  depth  → {dp_dir}")


if __name__ == "__main__":
    main()
