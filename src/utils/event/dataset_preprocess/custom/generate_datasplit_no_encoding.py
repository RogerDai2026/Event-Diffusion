# src/utils/data_split/generate_png_dataset_files.py

import numpy as np
import glob
import os
import re
from tqdm import tqdm


def extract_scene_and_frame(filename: str):
    """
    Extract scene and frame numbers from a filename of form 'scene{scene}_frame_{frame}.*'
    Returns (scene, frame) as ints, or (None, None) if not matched.
    """
    match = re.search(r'scene(\d+)_frame_(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def generate_png_dataset_files(
        root_dir: str = "/shared/qd8/data/output/",
        text_file_dir: str = "/home/qd8/code/Event-WassDiff/data_split/custom/",
        val_small_size: int = 900
):
    """
    Generate train/val/test splits and file lists for PNG event/depth pairs (no bin encoding).
    Scans each scene directory under root_dir, splits 43/6/6 scenes,
    then finds all .png in 'events' vs 'depth' subdirs by naming convention.
    Outputs custom_train.txt, custom_val.txt, custom_test.txt, and custom_val_small.txt
    with lines: '<event_rel_path> <depth_rel_path>'.
    """
    os.makedirs(text_file_dir, exist_ok=True)

    # Find scene directories
    patterns = [os.path.join(root_dir, 'scene_*'), os.path.join(root_dir, 'scene*')]
    scenes = sorted({d for pat in patterns for d in glob.glob(pat) if os.path.isdir(d)})
    print(f"Found {len(scenes)} scenes under {root_dir}.")
    if not scenes:
        return

    # Random split: 43 train, 6 val, 6 test
    np.random.seed(42)
    shuffled = scenes.copy()
    np.random.shuffle(shuffled)
    train_scenes = set(shuffled[:43])
    val_scenes = set(shuffled[43:49])
    test_scenes = set(shuffled[49:55])

    splits = {'train': [], 'val': [], 'test': []}

    # Process each scene
    for scene_path in tqdm(shuffled, desc="Processing scenes"):
        scene_name = os.path.basename(scene_path)
        if scene_path in train_scenes:
            split = 'train'
        elif scene_path in val_scenes:
            split = 'val'
        elif scene_path in test_scenes:
            split = 'test'
        else:
            continue

        # Identify event and depth dirs
        subdirs = [d for d in glob.glob(os.path.join(scene_path, '*')) if os.path.isdir(d)]
        # event dirs: any subdir not containing 'depth'
        event_dirs = [d for d in subdirs if 'depth' not in os.path.basename(d).lower()]
        # depth dirs: containing 'depth'
        depth_dirs = [d for d in subdirs if 'depth' in os.path.basename(d).lower()]
        if not event_dirs or not depth_dirs:
            print(f"Skipping {scene_name}: no event or depth dir")
            continue
        ev_dir = event_dirs[0]
        dp_dir = depth_dirs[0]

        # Collect .png files
        ev_files = glob.glob(os.path.join(ev_dir, '**', '*.npz'), recursive=True)
        dp_files = glob.glob(os.path.join(dp_dir, '**', '*.png'), recursive=True)

        # Map frame numbers to file paths
        ev_map = {extract_scene_and_frame(os.path.basename(f))[1]: f
                  for f in ev_files if extract_scene_and_frame(os.path.basename(f))[1] is not None}
        dp_map = {extract_scene_and_frame(os.path.basename(f))[1]: f
                  for f in dp_files if extract_scene_and_frame(os.path.basename(f))[1] is not None}

        common_frames = sorted(set(ev_map.keys()) & set(dp_map.keys()))
        for fr in common_frames:
            ev_rel = os.path.relpath(ev_map[fr], root_dir)
            dp_rel = os.path.relpath(dp_map[fr], root_dir)
            splits[split].append((ev_rel, dp_rel))

    # Shuffle within each split
    np.random.seed(42)
    for k in splits:
        np.random.shuffle(splits[k])

    # Write full split files
    for k in ['train', 'val', 'test']:
        out_path = os.path.join(text_file_dir, f"custom_{k}.txt")
        with open(out_path, 'w') as f:
            for ev, dp in splits[k]:
                f.write(f"{ev} {dp}\n")
        print(f"Wrote {len(splits[k])} entries to {out_path}")

    # Small validation subset
    val_list = splits['val']
    if val_list:
        small = val_list[:min(val_small_size, len(val_list))]
        out_small = os.path.join(text_file_dir, "custom_val_small.txt")
        with open(out_small, 'w') as f:
            for ev, dp in small:
                f.write(f"{ev} {dp}\n")
        print(f"Wrote {len(small)} entries to {out_small}")


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "/shared/qd8/data/output/"
    generate_png_dataset_files(root)
