#!/usr/bin/env python3
"""
Make E2Depth-style splits for MVSEC outdoor_day2 (NPZ inputs).

- vox-dir   : folder with per-frame voxel NPZs (key 'vox'), stems like 0000001234.npz
- depth-dir : folder with per-frame depth NPZs (key 'depth'), same stems
- Output    : src/data_split/mvsec/{train,val,test}.txt
Each line:  <abs_path_to_vox_npz> <abs_path_to_depth_npz>

Policy:
- Sort all matched stems numerically.
- Train = FIRST 8523 (contiguous).
- From the remaining, pick a RANDOM 1826 for val (once, with --seed), then keep order fixed (sorted).
- Test = the remaining 1826 (sorted).
"""

import argparse
from pathlib import Path
import re
import numpy as np

TRAIN_N = 5429
VAL_N   = 0
TEST_N  = 0

def numeric_stem(p: Path) -> int:
    m = re.match(r"^0*(\d+)$", p.stem)
    return int(m.group(1)) if m else int(p.stem)

def main():
    ap = argparse.ArgumentParser("MVSEC outdoor_day2 split (NPZ → train/val/test)")
    ap.add_argument("--vox-dir",   default="/shared/qd8/event3d/MVSEC/outdoor_night3_vox5",
                    help="Directory containing voxel NPZs (key 'vox').")
    ap.add_argument("--depth-dir", default="/shared/qd8/event3d/MVSEC/outdoor_night3_depth/left",
                    help="Directory containing depth NPZs (key 'depth').")
    ap.add_argument("--out-dir",   default="/home/qd8/code/Event-WassDiff/data_split/mvsec",
                    help="Where to write train.txt / val.txt / test.txt")
    ap.add_argument("--seed",      type=int, default=20240809,
                    help="Seed for the one-time random validation selection.")
    args = ap.parse_args()

    vox_dir = Path(args.vox_dir);  depth_dir = Path(args.depth_dir)
    out_dir = Path(args.out_dir);  out_dir.mkdir(parents=True, exist_ok=True)

    assert vox_dir.is_dir(),   f"vox-dir not found: {vox_dir}"
    assert depth_dir.is_dir(), f"depth-dir not found: {depth_dir}"

    # Collect .npz and match by stem
    vox_list   = sorted(vox_dir.glob("*.npz"),   key=numeric_stem)
    depth_list = sorted(depth_dir.glob("*.npz"), key=numeric_stem)

    vox_by_stem   = {p.stem: p for p in vox_list}
    depth_by_stem = {p.stem: p for p in depth_list}

    stems = sorted(set(vox_by_stem).intersection(depth_by_stem), key=lambda s: int(s))
    if not stems:
        raise RuntimeError("No overlapping stems between vox-dir and depth-dir.")
    needed = TRAIN_N + VAL_N + TEST_N
    if len(stems) < needed:
        raise RuntimeError(f"Found {len(stems)} matched pairs, but need {needed} (8523/1826/1826).")

    # 1) TRAIN: first 8523 (contiguous, fixed)
    stems_train = stems[:TRAIN_N]

    # 2) Remaining pool for val+test
    pool = stems[TRAIN_N:TRAIN_N + VAL_N + TEST_N]

    # 3) VAL: random 1826 from pool (once, with seed), then write in numeric order (stable eval)
    rng = np.random.RandomState(args.seed)
    val_idx = rng.choice(len(pool), size=VAL_N, replace=False)
    stems_val = sorted([pool[i] for i in val_idx], key=lambda s: int(s))

    # 4) TEST: the rest (sorted)
    mask = np.ones(len(pool), dtype=bool)
    mask[val_idx] = False
    leftover = [pool[i] for i in range(len(pool)) if mask[i]]
    stems_test = sorted(leftover[:TEST_N], key=lambda s: int(s))

    # Sanity checks
    assert len(stems_train) == TRAIN_N
    assert len(stems_val)   == VAL_N
    assert len(stems_test)  == TEST_N
    assert len(set(stems_train) & set(stems_val))  == 0
    assert len(set(stems_train) & set(stems_test)) == 0
    assert len(set(stems_val)   & set(stems_test)) == 0

    def write_list(stems, name):
        fpath = out_dir / f"{name}.txt"
        with fpath.open("w") as f:
            for s in stems:
                f.write(f"{vox_by_stem[s].resolve()} {depth_by_stem[s].resolve()}\n")
        return fpath

    f_train = write_list(stems_train, "outdoor_night3_testing")
    # f_val   = write_list(stems_val,   "val_1")
    # f_test  = write_list(stems_test,  "test_1")

    # print(f"[done] wrote:\n  {f_train}\n  {f_val}\n  {f_test}")
    print(f"[info] counts: train={len(stems_train)} val={len(stems_val)} test={len(stems_test)} (total matched={len(stems)})")

if __name__ == "__main__":
    main()
