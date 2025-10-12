#!/usr/bin/env python3
"""
Create final DSEC data splits with both event streams and NBIN3 encoding paths.

This creates split files in the format expected by the dataset loader:
nbin_3_encoding/sequence/eventfile.tif disparity_maps/sequence/disparity/image/depthfile.png
"""

import argparse
from pathlib import Path


def create_final_splits(nbin_root: Path, depth_root: Path, output_dir: Path):
    """Create final DSEC data splits with NBIN3 encoding paths."""
    
    # Define splits based on scenes
    train_sequences = [
        # Zurich sequences for training
        "zurich_city_00_a", "zurich_city_00_b", "zurich_city_01_a", "zurich_city_01_b", 
        "zurich_city_01_c", "zurich_city_01_d", "zurich_city_01_e", "zurich_city_01_f",
        "zurich_city_02_a", "zurich_city_02_b", "zurich_city_02_c", "zurich_city_02_d",
        "zurich_city_02_e", "zurich_city_03_a", "zurich_city_04_a", "zurich_city_04_b",
        "zurich_city_04_c", "zurich_city_04_d", "zurich_city_04_e", "zurich_city_04_f",
        "zurich_city_05_a", "zurich_city_05_b", "zurich_city_06_a", "zurich_city_07_a",
        "zurich_city_08_a", "zurich_city_09_a", "zurich_city_10_a", "zurich_city_10_b",
        "zurich_city_11_a", "zurich_city_11_b", "zurich_city_11_c"
    ]
    
    val_sequences = [
        # Interlaken sequences for validation
        "interlaken_00_c", "interlaken_00_d", "interlaken_00_e", "interlaken_00_f"
    ]
    
    test_sequences = [
        # Different scenes for testing
        "interlaken_00_g", "thun_00_a"
    ]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split_name, sequences in [("train", train_sequences), ("val", val_sequences), ("test", test_sequences)]:
        print(f"\nProcessing {split_name} split...")
        
        all_pairs = []
        for seq_name in sequences:
            print(f"  Processing {seq_name}...")
            
            nbin_dir = nbin_root / seq_name
            depth_dir = depth_root / seq_name / "disparity" / "image"
            
            if not nbin_dir.exists() or not depth_dir.exists():
                print(f"    Warning: Missing directories for {seq_name}")
                continue
            
            # Get NBIN files (should be named same as events: 0000000000.tif, etc.)
            nbin_files = sorted([f for f in nbin_dir.iterdir() if f.suffix == '.tif'])
            
            pairs = []
            for nbin_file in nbin_files:
                # Extract event index from NBIN filename
                try:
                    event_idx = int(nbin_file.stem)
                    # Map to corresponding depth index (2*event_idx)
                    depth_idx = 2 * event_idx
                    expected_depth_name = f"{depth_idx:06d}.png"
                    expected_depth_path = depth_dir / expected_depth_name
                    
                    if expected_depth_path.exists():
                        # Use relative paths for the split files
                        nbin_rel = f"nbin_3_encoding/{seq_name}/{nbin_file.name}"
                        depth_rel = f"disparity_maps/{seq_name}/disparity/image/{expected_depth_name}"
                        pairs.append((nbin_rel, depth_rel))
                    else:
                        print(f"    Warning: Missing depth file {expected_depth_name} for {seq_name}")
                except ValueError:
                    print(f"    Warning: Invalid NBIN filename {nbin_file.name} in {seq_name}")
                    continue
            
            all_pairs.extend(pairs)
            print(f"    Found {len(pairs)} pairs")
        
        # Write split file
        split_file = output_dir / f"{split_name}.txt"
        with open(split_file, 'w') as f:
            for nbin_path, depth_path in all_pairs:
                f.write(f"{nbin_path} {depth_path}\n")
        
        print(f"  Wrote {len(all_pairs)} pairs to {split_file}")
    
    print(f"\nCompleted! Final split files written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create final DSEC data splits with NBIN3 encoding")
    parser.add_argument("--nbin-root", type=Path, 
                        default="/shared/qd8/event3d/DSEC/train/nbin_3_encoding",
                        help="Root directory of NBIN3 encoded files")
    parser.add_argument("--depth-root", type=Path,
                        default="/shared/qd8/event3d/DSEC/train/disparity_maps", 
                        help="Root directory of depth maps")
    parser.add_argument("--output-dir", type=Path,
                        default="data_split/dsec_nbin3_final",
                        help="Output directory for final split files")
    
    args = parser.parse_args()
    
    print(f"NBIN3 root: {args.nbin_root}")
    print(f"Depth root: {args.depth_root}")
    print(f"Output dir: {args.output_dir}")
    
    create_final_splits(args.nbin_root, args.depth_root, args.output_dir)


if __name__ == "__main__":
    main()
