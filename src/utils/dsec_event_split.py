#!/usr/bin/env python3
import os, sys, time, argparse, csv, signal
from pathlib import Path
import numpy as np

# Make HDF5 behave on shared filesystems and load filters
os.environ.pop("HDF5_PLUGIN_PATH", None)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import hdf5plugin      # registers filters
import h5py
from tqdm.auto import tqdm

# ---------- Defaults (edit these if your paths move) ----------
DEFAULT_EVENTS_ROOT = Path("/shared/qd8/event3d/DSEC/train/events_raw")
DEFAULT_DISP_ROOT   = Path("/shared/qd8/event3d/DSEC/train/disparity_maps")
DEFAULT_OUT_ROOT    = Path("/shared/qd8/event3d/DSEC/train/event_streams")

# ---------- tiny helpers ----------
def log(msg): print(msg, flush=True)



def bisect_right_py(arr, x, lo=0, hi=None):
    # pure-Python binary search on a small 1D numpy slice
    if hi is None: hi = len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < arr[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo

def fast_searchsorted_bracketed(t_vec: np.ndarray, keys: np.ndarray, side="right", stride: int = 1024):
    """
    Robust searchsorted for huge 1D arrays.
    1) coarse on t_vec[::stride]
    2) refine in a small window around coarse*stride using pure-Python bisect.
    """
    assert t_vec.ndim == 1
    n = t_vec.shape[0]
    keys_q = keys.astype(t_vec.dtype, copy=False)

    # 1) coarse search on a tiny sampled array
    t_sample = t_vec[::stride]
    coarse = np.searchsorted(t_sample, keys_q, side=side)

    # 2) refine in a bounded local window (avoid big-array searchsorted)
    out = np.empty_like(coarse, dtype=np.int64)
    pad = 4 * stride  # window half-width
    for j, c in enumerate(coarse):
        approx = int(c) * stride
        start = max(0, approx - pad)
        end   = min(n, approx + pad)
        # refine with Python bisect on the small slice
        out[j] = start + bisect_right_py(t_vec[start:end], keys_q[j]) if side == "right" \
                 else start + (bisect_right_py(t_vec[start:end], keys_q[j]) - 1)
    return out


def timed(msg):
    class _T:
        def __enter__(self):
            self.t0 = time.time(); log(f"[T] {msg} …")
        def __exit__(self, *exc):
            log(f"[T] {msg} done in {time.time()-self.t0:.2f}s")
    return _T()

# ---------- EventSlicer-compatible helper ----------
class EventSlicer:
    """Same interface as official; caches t in RAM once."""
    def __init__(self, h5f):
        with timed("load events/t (timestamps)"):
            self.t_vec = h5f["events/t"][()]        # np.int64/uint32 monotonic
        self.x_ds = h5f["events/x"]
        self.y_ds = h5f["events/y"]
        self.p_ds = h5f["events/p"]

# ---------- Rectification ----------
def load_rectify_map(rect_h5: Path):
    with h5py.File(rect_h5, "r") as f:
        if "rectify_map" in f and isinstance(f["rectify_map"], h5py.Dataset) and f["rectify_map"].ndim == 3:
            return f["rectify_map"][()]            # (H,W,2) float32
        if "rectify_maps" in f:
            g = f["rectify_maps"]
            return np.stack([g["x"][()], g["y"][()]], axis=-1)
    raise KeyError(f"{rect_h5}: rectify_map not found")

def rectify_chunk(xi, yi, rect_map):
    H, W, _ = rect_map.shape
    xi = np.clip(xi, 0, W-1); yi = np.clip(yi, 0, H-1)
    xy = rect_map[yi, xi]                 # (N,2) float32
    return xy[:,0], xy[:,1]

# ---------- core per-sequence ----------
def process_sequence(seq_events_root: Path, seq_disp_root: Path, out_root: Path,
                     side: str, delta_ms: int, chunk: int, overwrite: bool,
                     compress: bool, limit: int, stride: int, no_rectify: bool,
                     only_index: int | None):
    seq = seq_events_root.name
    ev_dir  = seq_events_root / "events" / side
    ev_h5   = ev_dir / "events.h5"
    rect_h5 = ev_dir / "rectify_map.h5"

    disp_dir = seq_disp_root / "disparity"
    ts_txt   = disp_dir / "timestamps.txt"
    png_dir  = disp_dir / "event"

    if not (ev_h5.exists() and ts_txt.exists() and png_dir.exists()):
        log(f"[skip] {seq}: missing one of {ev_h5}, {ts_txt}, {png_dir}")
        return

    ts   = np.loadtxt(ts_txt, dtype=np.int64).reshape(-1)
    pngs = sorted(png_dir.glob("*.png"), key=lambda p: int(p.stem))
    if len(pngs) != ts.size:
        log(f"[warn] {seq}: len(pngs)={len(pngs)} != len(ts)={ts.size}; proceeding")

    assert int(pngs[0].stem) == 0, f"{seq}: first PNG should be 000000"
    ts, pngs = ts[1:], pngs[1:]                # official: drop first
    if stride > 1:
        ts, pngs = ts[::stride], pngs[::stride]
    if limit > 0:
        ts, pngs = ts[:limit], pngs[:limit]
    if only_index is not None:
        ts, pngs = ts[only_index:only_index+1], pngs[only_index:only_index+1]

    out_dir = out_root / seq / side
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_csv = out_dir / "pairs.csv"

    # graceful Ctrl-C
    def _stop(sig, frm): print("\nStopping…", flush=True); sys.exit(0)
    for s in (signal.SIGINT, signal.SIGTERM): signal.signal(s, _stop)

    delta_us = int(delta_ms) * 1000

    with h5py.File(ev_h5, "r") as h5f:
        slicer = EventSlicer(h5f)
        rect_map = None
        if not no_rectify:
            with timed("load rectify_map"):
                if rect_h5.exists():
                    rect_map = load_rectify_map(rect_h5)
                else:
                    log(f"[warn] {seq}: no rectify_map.h5; proceeding unrectified.")
        # precompute indices vectorized

        q_dtype = slicer.t_vec.dtype  # your t_vec showed uint32
        starts_q = (ts - delta_us).astype(q_dtype, copy=False)
        ts_q = ts.astype(q_dtype, copy=False)

        with timed("precompute slice indices (robust)"):
            i0_all = fast_searchsorted_bracketed(slicer.t_vec, starts_q, side="right", stride=1024)
            i1_all = fast_searchsorted_bracketed(slicer.t_vec, ts_q, side="right", stride=1024)

        print(f"first window indices: i0={int(i0_all[0])}, i1={int(i1_all[0])}, n={int(i1_all[0] - i0_all[0]):,}",
              flush=True)

        if i0_all.size:
            log(f"[{seq}] first window events ≈ {int(i1_all[0]-i0_all[0]):,}")

        with open(pairs_csv, "w", newline="") as fcsv, \
             tqdm(total=len(ts), desc=f"{seq}:{side}", unit="frame") as pbar:

            w = csv.writer(fcsv)
            w.writerow(["png_index","ts_start","ts_end","png_path","npz_path"])

            for k, (t_end, png_path, i0, i1) in enumerate(zip(ts, pngs, i0_all, i1_all)):
                t_start = int(t_end - delta_us)
                stem = png_path.stem
                out_npz = out_dir / f"{stem}.npz"

                if out_npz.exists() and not overwrite:
                    w.writerow([stem, t_start, int(t_end), str(png_path), str(out_npz)])
                    pbar.update(1); continue

                if i1 <= i0:
                    saver = np.savez_compressed if compress else np.savez
                    saver(out_npz, x=np.empty(0,np.float32), y=np.empty(0,np.float32),
                                  t=np.empty(0,np.int64),   p=np.empty(0,np.uint8))
                    w.writerow([stem, t_start, int(t_end), str(png_path), str(out_npz)])
                    pbar.update(1); continue

                # chunked read (and optional rectify)
                xs, ys, ts_out, ps = [], [], [], []
                n_total = i1 - i0
                for s in range(i0, i1, chunk):
                    e = min(s + chunk, i1)
                    with timed(f"read chunk [{s}:{e}) ({e-s:,} ev)"):
                        xi = slicer.x_ds[s:e].astype(np.int32,  copy=False)
                        yi = slicer.y_ds[s:e].astype(np.int32,  copy=False)
                        ti = slicer.t_vec[s:e]  # in RAM
                        pi = slicer.p_ds[s:e].astype(np.uint8,  copy=False)

                    if rect_map is not None:
                        with timed(f"rectify chunk ({e-s:,} ev)"):
                            xr, yr = rectify_chunk(xi, yi, rect_map)
                            xi = xr.astype(np.float32, copy=False)
                            yi = yr.astype(np.float32, copy=False)
                    else:
                        xi = xi.astype(np.float32, copy=False)
                        yi = yi.astype(np.float32, copy=False)

                    xs.append(xi); ys.append(yi); ts_out.append(ti); ps.append(pi)

                with timed(f"concat & save ({n_total:,} ev total)"):
                    x_cat = np.concatenate(xs) if xs else np.empty(0, np.float32)
                    y_cat = np.concatenate(ys) if ys else np.empty(0, np.float32)
                    t_cat = np.concatenate(ts_out) if ts_out else np.empty(0, np.int64)
                    p_cat = np.concatenate(ps) if ps else np.empty(0, np.uint8)
                    saver = np.savez_compressed if compress else np.savez
                    saver(out_npz, x=x_cat, y=y_cat, t=t_cat, p=p_cat)

                w.writerow([stem, t_start, int(t_end), str(png_path), str(out_npz)])
                pbar.update(1)

    log(f"[ok] {seq}: wrote → {out_dir}")

# ---------- CLI / defaults ----------
def main():
    ap = argparse.ArgumentParser(description="Fast OFFICIAL-style DSEC slicer (Δ window, chunked I/O)")
    ap.add_argument("--events-root", default=str(DEFAULT_EVENTS_ROOT),
                    help="root with <seq>/events/<side>/{events.h5,rectify_map.h5}")
    ap.add_argument("--disp-root",   default=str(DEFAULT_DISP_ROOT),
                    help="root with <seq>/disparity/{timestamps.txt,event/*.png}")
    ap.add_argument("--out-root",    default=str(DEFAULT_OUT_ROOT),
                    help="where to write event_streams")
    ap.add_argument("--side", choices=["left","right"], default="left")
    ap.add_argument("--delta-ms", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=2_000_000, help="events per chunk")
    ap.add_argument("--limit", type=int, default=0, help="max frames per seq (0=all)  [default=10 for debug]")
    ap.add_argument("--stride", type=int, default=1,  help="keep every K-th frame")
    ap.add_argument("--seq", default="", help="process only this sequence name (exact match)")
    ap.add_argument("--index", type=int, default=-1, help="process only this frame index after dropping first (>=0)")
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-rectify", action="store_true")
    args = ap.parse_args()

    events_root = Path(args.events_root)
    disp_root   = Path(args.disp_root)
    out_root    = Path(args.out_root)

    seqs = sorted([p for p in events_root.iterdir() if p.is_dir()])
    if args.seq:
        seqs = [events_root / args.seq]
        if not seqs[0].exists():
            log(f"[err] sequence {args.seq} not found under {events_root}"); sys.exit(1)

    for seq_ev in tqdm(seqs, desc="Sequences", unit="seq"):
        seq_dp = disp_root / seq_ev.name
        only_index = None if args.index < 0 else args.index
        process_sequence(seq_ev, seq_dp, out_root,
                         args.side, args.delta_ms, args.chunk,
                         args.overwrite, args.compress,
                         args.limit, args.stride, args.no_rectify,
                         only_index)

if __name__ == "__main__":
    main()
