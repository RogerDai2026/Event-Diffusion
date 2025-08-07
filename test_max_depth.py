#!/usr/bin/env python3
import os
import argparse
import numpy as np
from PIL import Image
import glob
import tqdm

def decode_depth_from_rgb(depth_array):
    """
    Given HxW x 3 RGB depth image (Carla-style), decode to meters.
    """
    R = depth_array[:, :, 0].astype(np.float64)
    G = depth_array[:, :, 1].astype(np.float64)
    B = depth_array[:, :, 2].astype(np.float64)
    normalized = (R + G * 256 + B * 256 * 256) / (256**3 - 1)
    depth_meters = 1000.0 * normalized
    return depth_meters

def load_depth(path):
    arr = np.array(Image.open(path))
    if arr.ndim == 3 and arr.shape[2] >= 3:
        # RGB encoding
        return decode_depth_from_rgb(arr[:, :, :3])
    else:
        # single-channel, assume already in meters or normalized; user can adjust
        return arr.astype(np.float64)

def sample_valid_depths(depth_paths, max_samples_per_file=1000, random_seed=42):
    rng = np.random.default_rng(random_seed)
    all_vals = []
    for p in tqdm.tqdm(depth_paths, desc="Loading depth files"):
        try:
            d = load_depth(p)
        except Exception as e:
            print(f"  [warning] failed to read {p}: {e}")
            continue
        # keep only positive depths
        valid = d > 0
        if not np.any(valid):
            continue
        vals = d[valid].flatten()
        if vals.size > max_samples_per_file:
            idx = rng.choice(vals.size, size=max_samples_per_file, replace=False)
            vals = vals[idx]
        all_vals.append(vals)
    if not all_vals:
        raise RuntimeError("No valid depth values found.")
    all_vals = np.concatenate(all_vals, axis=0)
    return all_vals

def main():
    parser = argparse.ArgumentParser(description="Compute depth percentiles for Carla depth dataset")
    parser.add_argument("--depth_glob", type=str, default="/shared/qd8/data/output/scene_45/depth/*.png",
                        help="Glob pattern to find depth files, e.g. '/path/to/*/depth/*.png'")
    parser.add_argument("--max_samples_per_file", type=int, default=1000,
                        help="How many valid depth pixels to sample per file to keep memory reasonable.")
    parser.add_argument("--percentiles", type=float, nargs="+", default=[90, 95, 99, 100],
                        help="Percentiles to compute.")
    parser.add_argument("--save_histogram", action="store_true",
                        help="If set, dump a histogram of depths to npz (depth_hist.npz).")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.depth_glob, recursive=True))
    if not paths:
        print("No depth files found for pattern:", args.depth_glob)
        return
    print(f"Found {len(paths)} depth files. Sampling up to {args.max_samples_per_file} valid depths per file.")

    all_depths = sample_valid_depths(paths, max_samples_per_file=args.max_samples_per_file)

    # Filter out non-finite
    all_depths = all_depths[np.isfinite(all_depths)]
    print(f"Total sampled valid depth values: {all_depths.size}")

    # Compute stats
    pcts = np.percentile(all_depths, args.percentiles)
    for pct, val in zip(args.percentiles, pcts):
        print(f"  {pct:.1f}th percentile: {val:.4f} meters")
    print(f"  min: {all_depths.min():.4f}, max: {all_depths.max():.4f}")
    mean = np.mean(all_depths)
    std = np.std(all_depths)
    print(f"  mean: {mean:.4f}, std: {std:.4f}")

    if args.save_histogram:
        hist, bin_edges = np.histogram(all_depths, bins=200, density=True)
        np.savez("depth_hist.npz", hist=hist, bin_edges=bin_edges)
        print("Saved histogram to depth_hist.npz")

    recommended_max = np.percentile(all_depths, 99)
    print(f"\nSuggested max_depth (99th percentile): {recommended_max:.4f} meters")

if __name__ == "__main__":
    main()