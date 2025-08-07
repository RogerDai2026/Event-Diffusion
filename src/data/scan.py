#!/usr/bin/env python3
import os
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

# allow truncated PNGs
ImageFile.LOAD_TRUNCATED_IMAGES = True

def check_dataset(filename_ls_path: str, dataset_dir: str):
    bad = []
    with open(filename_ls_path, "r") as f:
        lines = [l.strip().split() for l in f if l.strip()]

    for i, (ev_rel, depth_rel) in enumerate(tqdm(lines, desc="Scanning dataset", unit="sample")):
        ev_path    = os.path.join(dataset_dir, ev_rel)
        depth_path = os.path.join(dataset_dir, depth_rel)

        # 1) Check event file (.npz)
        try:
            if not os.path.exists(ev_path):
                raise ValueError("file not found")
            if os.path.getsize(ev_path) == 0:
                raise ValueError("zero-length file")
            data = np.load(ev_path)
            keys = list(data.keys())
            if not keys or data[keys[0]].size == 0:
                raise ValueError("empty array inside npz")
        except Exception as e:
            bad.append((i, ev_rel, f"event error: {e}"))
            continue

        # 2) Check depth file (PNG)
        try:
            if not os.path.exists(depth_path):
                raise ValueError("file not found")
            with Image.open(depth_path) as img:
                img.verify()
                img = Image.open(depth_path)
                _ = np.array(img)
        except Exception as e:
            bad.append((i, depth_rel, f"depth error: {e}"))

    if not bad:
        print("✅ All files look good!")
    else:
        print("❌ Found problematic files:")
        for idx, rel, reason in bad:
            print(f"  [{idx:4d}] {rel:40s} ⇒ {reason}")


def main():
    filenames = "/home/qdai/code/Event-WassDiff/data_split/custom_nbins/custom_val_small.txt"
    base_dir  = "/scratch/shared/data/output/"

    total = len(open(filenames).readlines())
    print(f"Scanning {total} samples…\n")
    check_dataset(filenames, base_dir)


if __name__ == "__main__":
    main()
