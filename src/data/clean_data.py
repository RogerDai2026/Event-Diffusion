#!/usr/bin/env python3
import os

def clean_and_prune(train_file: str, base_dir: str, bad: set):
    # Load all lines
    with open(train_file, "r") as f:
        lines = f.readlines()

    clean_lines = []
    removed_count = 0

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 2:
            # keep malformed or blank lines
            clean_lines.append(line)
            continue

        ev_rel, depth_rel = parts
        if ev_rel in bad or depth_rel in bad:
            removed_count += 1
            # attempt to delete both files
            for rel in (ev_rel, depth_rel):
                path = os.path.join(base_dir, rel)
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                        print(f"Deleted file: {path}")
                    except Exception as e:
                        print(f"⚠️ Could not delete {path}: {e}")
            # skip writing this line
        else:
            clean_lines.append(line)

    # Overwrite the original train_file
    with open(train_file, "w") as f:
        f.writelines(clean_lines)

    print(f"\nDone. Removed {removed_count} samples (and their files).")


def main():
    # 1) Paths – update these if needed
    train_file = "/home/qdai/code/Event-WassDiff/data_split/custom/custom_val_small.txt"
    base_dir   = "/scratch/shared/data/output/"

    # 2) The exact relative paths of any known bad files: for train
    # bad = {
    #     "scene_32/events/dvs_scene32_frame_025288.npz",
    #     "scene_03/events/dvs_scene03_frame_001589.npz",
    #     "scene_03/events/dvs_scene03_frame_001581.npz",
    #     "scene_03/events/dvs_scene03_frame_001583.npz",
    #     "scene_03/events/dvs_scene03_frame_001579.npz",
    #     "scene_03/events/dvs_scene03_frame_001564.npz",
    #     "scene_03/events/dvs_scene03_frame_001573.npz",
    #     "scene_41/depth/depth_scene40_frame_013231.png",
    #     "scene_29/depth/depth_scene29_frame_002367.png",
    #     "scene_30/depth/depth_scene30_frame_017368.png",
    #     "scene_07/events/dvs_scene07_frame_003540.npz",
    #     "scene_34/events/dvs_scene34_frame_003454.npz",
    #     "scene_03/events/dvs_scene03_frame_001582.npz",
    #     "scene_55/events/dvs_scene55_frame_002278.npz",
    #     "scene_03/events/dvs_scene03_frame_001590.npz",
    #     "scene_03/events/dvs_scene03_frame_001571.npz",
    #     "scene_03/events/dvs_scene03_frame_001587.npz",
    # }

    # for validation
    bad = {
    "scene_03 / events / dvs_scene03_frame_001565.npz",
    "scene_55 / events / dvs_scene55_frame_002279.npz",
    "scene_03 / events / dvs_scene03_frame_001588.npz",
    "scene_50 / events / dvs_scene50_frame_005838.npz",
    }

    print(f"Cleaning {train_file} under base dir {base_dir}…")
    clean_and_prune(train_file, base_dir, bad)


if __name__ == "__main__":
    main()
