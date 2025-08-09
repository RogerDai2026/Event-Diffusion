#!/usr/bin/env python3
"""
Script to generate NBIN3 encodings from event NPZ files,
with a check that each output is exactly 720×1280 before saving.
"""
import os
import glob
import numpy as np
import imageio.v3 as io
from tqdm import tqdm
import argparse
from numpy import ndarray


def normalize(x: ndarray,
              relative_vmin: float = None,
              relative_vmax: float = None,
              interval_vmax: float = None) -> ndarray:
    """
    Normalize an array x into [0,1], optionally using relative bounds.
    """
    vmax, vmin = float(x.max()), float(x.min())
    if relative_vmax is not None:
        vmax = relative_vmax + vmin
    if relative_vmin is not None:
        vmin = relative_vmin + vmin
    if interval_vmax is None:
        interval_vmax = vmax - vmin
    # Clip values
    x = x * (x >= vmin) * (x <= vmax)
    return (x - vmin) / interval_vmax


def nbin_encoding(
    times: ndarray,
    polarity: ndarray,
    x: ndarray,
    y: ndarray,
    height: int,
    width: int,
    nbin: int = 3
) -> ndarray:
    """
    Generate NBIN encoding from event data.
    times, polarity, x, y are 1D arrays of the same length.
    height, width are the desired frame dims.
    nbin is the number of temporal bins.
    """
    x = x.astype(np.int64)
    y = y.astype(np.int64)

    # Normalize timestamps into [0,1]
    normalized_time = normalize(times)

    # Convert polarity to 0 / 255
    polarity_enc = polarity.copy()
    polarity_enc[polarity_enc == 1]  = 255
    polarity_enc[polarity_enc == -1] = 0
    # any 0/True/False -> 0/255
    polarity_enc[polarity_enc == 0]     = 0
    polarity_enc[polarity_enc == True]  = 255
    polarity_enc[polarity_enc == False] = 0

    # allocate output
    encoded = np.ones((nbin, height, width), dtype=np.uint8) * 128

    # assign each event to a time‐bin
    time_bin = np.minimum((normalized_time * nbin).astype(int), nbin - 1)

    for b in range(nbin):
        frame = np.ones((height, width), dtype=np.uint8) * 128
        mask  = (time_bin == b)
        if np.any(mask):
            # bounds check
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


def process_custom_dataset(base_dir: str, output_dir: str):
    """Process all scenes in the custom dataset, enforce 720×1280 outputs."""
    print(f"Processing custom dataset from: {base_dir}")
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    scene_dirs = sorted(glob.glob(os.path.join(base_dir, "scene*")))
    print(f"Found {len(scene_dirs)} scene directories")
    total_processed = 0

    for scene_dir in tqdm(scene_dirs, desc="Scenes"):
        scene_name = os.path.basename(scene_dir)
        events_dir = os.path.join(scene_dir, "events")
        if not os.path.isdir(events_dir):
            print(f"  Warning: no events dir in {scene_name}, skipping")
            continue

        npz_files = sorted(glob.glob(os.path.join(events_dir, "*.npz")))
        print(f"  {scene_name}: found {len(npz_files)} NPZ files")
        if not npz_files:
            continue

        scene_out = os.path.join(output_dir, scene_name, "N_BINS_3")
        os.makedirs(scene_out, exist_ok=True)

        for npz_file in tqdm(npz_files, desc=f"  {scene_name}", leave=False):
            try:
                data = np.load(npz_file)
                # extract events array
                events = data['events'] if 'events' in data else data[list(data.keys())[0]]

                # parse structured vs. plain
                if events.dtype.names:
                    x   = events['x'].astype(int)
                    y   = events['y'].astype(int)
                    t   = events['t'].astype(float)
                    pol = events['pol']
                    if pol.dtype == bool:
                        pol = pol.astype(int)*2 - 1
                    elif np.all(np.isin(pol, [0,1])):
                        pol = pol*2 - 1
                else:
                    t, x, y, pol = (
                        events[:,0].astype(float),
                        events[:,1].astype(int),
                        events[:,2].astype(int),
                        events[:,3].astype(int),
                    )
                    if np.all(np.isin(pol, [0,1])):
                        pol = pol*2 - 1

                if len(x) == 0:
                    print(f"    Warning: no events in {npz_file}, skipping")
                    continue

                # force sensor resolution
                H_raw, W_raw = 720, 1280

                # do encoding
                encoded = nbin_encoding(t, pol, x, y, H_raw, W_raw, nbin=3)

                # sanity‐check
                _, H_out, W_out = encoded.shape
                if (H_out, W_out) != (720, 1280):
                    print(f"    [skip] {os.path.basename(npz_file)}: got {(H_out, W_out)}, expected (720,1280)")
                    continue

                # save as TIFF
                name     = os.path.splitext(os.path.basename(npz_file))[0]
                out_path = os.path.join(scene_out, f"{name}.tif")
                io.imwrite(out_path, encoded)
                total_processed += 1

            except Exception as e:
                print(f"    Error on {npz_file}: {e}")
                continue

    print(f"\n✅ Done: generated {total_processed} NBIN3 files (720×1280) in {output_dir}")


def main():
    parser = argparse.ArgumentParser("NBIN3 encoder with size check")
    parser.add_argument("--base_dir",  type=str, default="/shared/qd8/data/output/")
    parser.add_argument("--output_dir", type=str, default="/shared/qd8/data/output/")
    args = parser.parse_args()
    process_custom_dataset(args.base_dir, args.output_dir)


if __name__ == "__main__":
    main()
