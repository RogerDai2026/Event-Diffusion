#!/usr/bin/env python3
"""
Click-to-run MVSEC ROS bag → NPZ exporter (sudo-free, explicit typestore for rosbags).

What it does:
  • Reads Outdoor Night 2 & 3 ROS1 .bag files from BASE_DIR
  • Uses rosbags with an explicit typestore (no ROS install required)
  • Exports per-depth-frame NPZs:
      events_out/0000000000.npz : x[int16], y[int16], t[float64], p[int8∈{-1,0,+1}]
      depth_out/0000000000.npz  : depth[float32], shape (260,346)
  • Default windowing: centered 50 ms around each depth timestamp

Requires (in your user env):
  python -m pip install rosbags numpy tqdm
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Any

import numpy as np

# ---- Optional progress bar (safe fallback if tqdm not installed) ----
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # type: ignore
        return it

# ---- rosbags (pure Python) ----
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


# ======================
# ==== CONFIG HERE  ====
# ======================

BASE_DIR = Path("/shared/qd8/event3d/MVSEC").expanduser()
BAGS_DIR = BASE_DIR  # bags are directly in BASE_DIR

SEQUENCES = [
    {
        "name": "outdoor_night2",
        "data_bag": BAGS_DIR / "outdoor_night2_data.bag",
        "gt_bag":   BAGS_DIR / "outdoor_night2_gt.bag",
        "events_out": BASE_DIR / "outdoor_night2_events" / "left",
        "depth_out":  BASE_DIR / "outdoor_night2_depth" / "left",
    },
    {
        "name": "outdoor_night3",
        "data_bag": BAGS_DIR / "outdoor_night3_data.bag",
        "gt_bag":   BAGS_DIR / "outdoor_night3_gt.bag",
        "events_out": BASE_DIR / "outdoor_night3_events" / "left",
        "depth_out":  BASE_DIR / "outdoor_night3_depth" / "left",
    },
]

SIDE       = "left"       # "left" | "right"
MODE       = "centered"   # "centered" | "between"
WINDOW_MS  = 50.0
START_IDX  = 0
LIMIT      = 0            # 0 = all frames
POL_SCHEME = "zpm1"       # "zpm1" | "pm1" | "01"
OVERWRITE  = False
PRINT_TOPICS = True       # dump bag topics once for sanity


# ======================
# ====== HELPERS =======
# ======================

# Inlined message definitions for dvs_msgs (no ROS install needed)
EVENT_MSG = """\
int16 x
int16 y
time ts
bool polarity
"""
EVENTARRAY_MSG = """\
std_msgs/Header header
dvs_msgs/Event[] events
"""

def build_typestore():
    """Build ROS1 typestore and register custom dvs_msgs types."""
    typestore = get_typestore(Stores.ROS1_NOETIC)
    add_types = {}
    add_types.update(get_types_from_msg(EVENT_MSG, 'dvs_msgs/msg/Event'))
    add_types.update(get_types_from_msg(EVENTARRAY_MSG, 'dvs_msgs/msg/EventArray'))
    typestore.register(add_types)
    return typestore

def list_topics(bag_path: Path, typestore) -> None:
    """Print available topics and msgtypes in a bag (debug helper)."""
    try:
        with AnyReader([bag_path], default_typestore=typestore) as reader:
            print(f"[topics] {bag_path.name}")
            seen = set()
            for c in reader.connections:
                key = (c.topic, c.msgtype)
                if key not in seen:
                    print(f"  - {c.topic} :: {c.msgtype}")
                    seen.add(key)
    except Exception as e:
        print(f"[warn] Could not list topics for {bag_path}: {e}")

def to_int8_pol(p: np.ndarray, scheme: str = POL_SCHEME) -> np.ndarray:
    """Normalize polarity to int8: zpm1 (default), pm1, or 01."""
    p = np.asarray(p)
    if p.dtype == np.bool_:
        p = p.astype(np.int8)
        if scheme == "pm1":
            return (2 * p - 1).astype(np.int8)  # False→-1, True→+1
        elif scheme == "01":
            return p  # False→0, True→1
        else:  # zmp1 (default)
            return (2 * p - 1).astype(np.int8)  # False→-1, True→+1
    
    p = p.astype(np.int8, copy=False)
    if scheme == "pm1":
        if np.min(p) >= 0:
            return (2 * p - 1).astype(np.int8)
        return np.where(p == 0, -1, np.sign(p)).astype(np.int8)
    if scheme == "01":
        return np.clip(p, 0, 1).astype(np.int8)
    return np.clip(np.sign(p), -1, 1).astype(np.int8)

def image_msg_to_numpy(img) -> np.ndarray:
    """sensor_msgs/Image (32FC1) -> (H,W) float32 ndarray."""
    H, W = img.height, img.width
    arr = np.frombuffer(img.data, dtype=np.float32).reshape(H, W)
    if getattr(img, "is_bigendian", 0):
        arr = arr.byteswap().newbyteorder()
    return arr

def ros_time_to_sec(t: Any) -> float:
    """
    Robustly convert ROS time to seconds.
    Supports:
      - ROS1:  .sec + .nsec
      - ROS2:  .sec + .nanosec (common in rosbags deserialization)
      - numeric: float/int seconds already
    """
    # numeric input
    if isinstance(t, (int, float, np.floating)):
        return float(t)
    # objects with fields
    sec = getattr(t, 'sec', None)
    if sec is None:
        # sometimes header.stamp might be nested or dict-like; last resort:
        try:
            return float(t)
        except Exception:
            raise AttributeError(f"Unsupported time object: {type(t)} with attrs {dir(t)}")
    # nsec / nanosec
    nsec = getattr(t, 'nsec', None)
    if nsec is None:
        nsec = getattr(t, 'nanosec', None)
    if nsec is None:
        # some builds may use 'ns'
        nsec = getattr(t, 'ns', 0)
    return float(sec) + float(nsec) * 1e-9


def export_sequence(
    data_bag: Path,
    gt_bag: Path,
    events_out: Path,
    depth_out: Path,
    typestore,
    side: str = SIDE,
    mode: str = MODE,
    window_ms: float = WINDOW_MS,
    start_idx: int = START_IDX,
    limit: int = LIMIT,
    pol_scheme: str = POL_SCHEME,
    overwrite: bool = OVERWRITE,
) -> None:
    """Process one (data, gt) pair into per-depth-frame NPZs."""
    topic_ev = f"/davis/{side}/events"
    topic_dp = f"/davis/{side}/depth_image_rect"

    events_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    # ---- Pass 1: collect depth timestamps & write depth NPZs
    depth_ts: List[float] = []
    with AnyReader([gt_bag], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == topic_dp]
        if not conns:
            raise RuntimeError(f"No depth topic {topic_dp} in {gt_bag}")
        conn = conns[0]

        k = 0
        H = W = None
        for _conn, _t, raw in tqdm(reader.messages(connections=[conn]), desc=f"{gt_bag.name}: depth", unit="frm"):
            msg = reader.deserialize(raw, conn.msgtype)
            if k == 0:
                H, W = msg.height, msg.width
                assert (H, W) == (260, 346), f"Expected 260x346, got {H}x{W}"
            depth_ts.append(ros_time_to_sec(msg.header.stamp))

            dp_path = depth_out / f"{k:010d}.npz"
            if overwrite or (not dp_path.exists()):
                depth = image_msg_to_numpy(msg).astype(np.float32, copy=False)
                np.savez_compressed(dp_path, depth=depth)
            k += 1

    depth_ts = np.asarray(depth_ts, dtype=np.float64)
    N = len(depth_ts)
    if limit > 0:
        end_idx = min(start_idx + limit, N)
    else:
        end_idx = N
    start_idx = max(start_idx, 0)

    # Build time windows
    if mode == "between":
        # edges: (t_{k-1}, t_k]
        pass
    else:
        half = 0.5 * (window_ms * 1e-3)
        left_edge  = depth_ts - half
        right_edge = depth_ts + half

    # ---- Pass 2: stream events and write per-window event NPZs
    with AnyReader([data_bag], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == topic_ev]
        if not conns:
            raise RuntimeError(f"No event topic {topic_ev} in {data_bag}")
        conn = conns[0]

        k = start_idx
        xs: List[int] = []
        ys: List[int] = []
        ts: List[float] = []
        ps: List[bool] = []

        if mode == "between":
            prev_edge = depth_ts[k - 1] if k > 0 else -np.inf
            right = depth_ts[k]
        else:
            left = left_edge[k]
            right = right_edge[k]

        for _conn, _t, raw in tqdm(reader.messages(connections=[conn]), desc=f"{data_bag.name}: events", unit="msg"):
            msg = reader.deserialize(raw, conn.msgtype)  # dvs_msgs/EventArray
            if not msg.events:
                continue

            m = len(msg.events)
            mx = np.fromiter((e.x for e in msg.events), dtype=np.int16, count=m)
            my = np.fromiter((e.y for e in msg.events), dtype=np.int16, count=m)
            mt = np.fromiter((ros_time_to_sec(e.ts) for e in msg.events), dtype=np.float64, count=m)
            mp = np.fromiter((e.polarity for e in msg.events), dtype=np.bool_, count=m)

            i = 0
            while i < m and k < end_idx:
                # If we reached/passed the window's right edge, flush current window
                if mt[i] >= right:
                    stem = f"{k:010d}"
                    ev_path = events_out / f"{stem}.npz"
                    if overwrite or (not ev_path.exists()):
                        if ts:
                            x = np.asarray(xs, dtype=np.int16)
                            y = np.asarray(ys, dtype=np.int16)
                            t = np.asarray(ts, dtype=np.float64)
                            p = to_int8_pol(np.asarray(ps), scheme=pol_scheme)
                            if x.size:
                                if x.max() >= 346 or y.max() >= 260 or x.min() < 0 or y.min() < 0:
                                    raise ValueError(
                                        f"Out-of-bounds at frame {k}: "
                                        f"x[{x.min()},{x.max()}], y[{y.min()},{y.max()}] with W=346,H=260"
                                    )
                            np.savez_compressed(ev_path, x=x, y=y, t=t, p=p)
                        else:
                            np.savez_compressed(
                                ev_path,
                                x=np.empty((0,), dtype=np.int16),
                                y=np.empty((0,), dtype=np.int16),
                                t=np.empty((0,), dtype=np.float64),
                                p=np.empty((0,), dtype=np.int8),
                            )
                    # advance window
                    k += 1
                    xs.clear(); ys.clear(); ts.clear(); ps.clear()
                    if k >= end_idx:
                        break
                    if mode == "between":
                        prev_edge = depth_ts[k - 1] if k > 0 else -np.inf
                        right = depth_ts[k]
                    else:
                        left = left_edge[k]; right = right_edge[k]
                    continue

                # Otherwise, consume event if inside current window
                inside = (mt[i] > prev_edge and mt[i] <= right) if mode == "between" \
                         else (mt[i] >= left and mt[i] < right)
                if inside:
                    xs.append(int(mx[i])); ys.append(int(my[i])); ts.append(float(mt[i])); ps.append(bool(mp[i]))
                i += 1

        # Flush any remaining windows if no further events advanced time
        while k < end_idx:
            stem = f"{k:010d}"
            ev_path = events_out / f"{stem}.npz"
            if overwrite or (not ev_path.exists()):
                if ts:
                    x = np.asarray(xs, dtype=np.int16)
                    y = np.asarray(ys, dtype=np.int16)
                    t = np.asarray(ts, dtype=np.float64)
                    p = to_int8_pol(np.asarray(ps), scheme=pol_scheme)
                    np.savez_compressed(ev_path, x=x, y=y, t=t, p=p)
                else:
                    np.savez_compressed(
                        ev_path,
                        x=np.empty((0,), dtype=np.int16),
                        y=np.empty((0,), dtype=np.int16),
                        t=np.empty((0,), dtype=np.float64),
                        p=np.empty((0,), dtype=np.int8),
                    )
            k += 1
            xs.clear(); ys.clear(); ts.clear(); ps.clear()

    print(f"[done] events → {events_out}\n[done] depth  → {depth_out}")


# ======================
# ===== ENTRYPOINT =====
# ======================

def main():
    print("=== MVSEC ROS bag → NPZ exporter (sudo-free) ===")
    print(f"Base dir: {BASE_DIR}")
    print(f"Side: {SIDE} | Mode: {MODE} | Window(ms): {WINDOW_MS} | Start: {START_IDX} | Limit: {LIMIT or 'ALL'}")

    typestore = build_typestore()

    for seq in SEQUENCES:
        name = seq["name"]
        data_bag = Path(seq["data_bag"])
        gt_bag   = Path(seq["gt_bag"])

        print(f"\n--- {name} ---")
        if not data_bag.exists():
            raise FileNotFoundError(f"Missing bag: {data_bag}")
        if not gt_bag.exists():
            raise FileNotFoundError(f"Missing bag: {gt_bag}")

        if PRINT_TOPICS:
            list_topics(data_bag, typestore)
            list_topics(gt_bag, typestore)

        export_sequence(
            data_bag=data_bag,
            gt_bag=gt_bag,
            events_out=seq["events_out"],
            depth_out=seq["depth_out"],
            typestore=typestore,
        )

    print("\n[all done]")


if __name__ == "__main__":
    main()
