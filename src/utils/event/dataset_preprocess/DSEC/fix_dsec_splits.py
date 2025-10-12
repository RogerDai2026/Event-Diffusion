#!/usr/bin/env python3
"""
Fix DSEC data splits to correctly map event indices to depth indices.

The issue:
- Events are named: 0000000000.npz, 0000000001.npz, ... (sequential)
- Depth images are named: 000000.png, 000002.png, 000004.png, ... (every 2nd frame)
- Event index N corresponds to depth index 2*N

This script creates corrected data split files.
"""

import argparse
from pathlib import Path
from typing import List, Tuple


def get_sequence_mapping(events_root: Path, depth_root: Path, seq_name: str) -> List[Tuple[str, str]]:
    """
    Get the correct event->depth mapping for a sequence.
    
    Returns:
        List of (event_path, depth_path) tuples
    """
    events_dir = events_root / seq_name
    depth_dir = depth_root / seq_name / "disparity" / "image"
    
    if not events_dir.exists() or not depth_dir.exists():
        print(f"Warning: Missing directories for {seq_name}")
        return []
    
    # Get sorted event files
    event_files = sorted([f for f in events_dir.iterdir() if f.suffix == '.npz'])
    
    # Get sorted depth files  
    depth_files = sorted([f for f in depth_dir.iterdir() if f.suffix == '.png'])
    
    # Create mapping
    pairs = []
    for i, event_file in enumerate(event_files):
        # Event index i should map to depth index 2*i
        expected_depth_name = f"{2*i:06d}.png"
        expected_depth_path = depth_dir / expected_depth_name
        
        if expected_depth_path.exists():
            # Use relative paths for the split files
            event_rel = f"event_streams/{seq_name}/{event_file.name}"
            depth_rel = f"disparity_maps/{seq_name}/disparity/image/{expected_depth_name}"
            pairs.append((event_rel, depth_rel))
        else:
            print(f"Warning: Missing depth file {expected_depth_name} for {seq_name}")
    
    return pairs


def create_dsec_splits(events_root: Path, depth_root: Path, output_dir: Path):
    """Create corrected DSEC data splits."""
    
    # Define splits based on scenes (as originally planned)
    train_sequences = [
        # Zurich sequences for training
        "zurich_city_00_a", "zurich_city_00_b", "zurich_city_01_a", "zurich_city_01_b", 
        "zurich_city_01_c", "zurich_city_01_d", "zurich_city_01_e", "zurich_city_01_f",
        "zurich_city_02_a", "zurich_city_02_b", "zurich_city_02_c", "zurich_city_02_d",
        "zurich_city_02_e", "zurich_city_03_a", "zurich_city_04_a", "zurich_city_04_b",
        "zurich_city_04_c", "zurich_city_04_d", "zurich_city_04_e", "zurich_city_04_f",
        "zurich_city_05_a", "zurich_city_05_b", "zurich_city_06_a", "zurich_city_07_a",
        "zurich_city_08_a", "zurich_city_09_a", "zurich_city_10_a", "zurich_city_10_b",
        "zurich_city_11_a", "zurich_city_11_b", "zurich_city_11_c", "zurich_city_12_a",
        "zurich_city_13_a", "zurich_city_14_a", "zurich_city_15_a"
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
            pairs = get_sequence_mapping(events_root, depth_root, seq_name)
            all_pairs.extend(pairs)
            print(f"    Found {len(pairs)} pairs")
        
        # Write split file
        split_file = output_dir / f"{split_name}.txt"
        with open(split_file, 'w') as f:
            for event_path, depth_path in all_pairs:
                f.write(f"{event_path} {depth_path}\n")
        
        print(f"  Wrote {len(all_pairs)} pairs to {split_file}")
    
    print(f"\nCompleted! Split files written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fix DSEC data splits with correct event->depth mapping")
    parser.add_argument("--events-root", type=Path, 
                        default="/shared/qd8/event3d/DSEC/train/nbin_3_encoding",
                        help="Root directory of extracted events")
    parser.add_argument("--depth-root", type=Path,
                        default="/shared/qd8/event3d/DSEC/train/disparity_maps", 
                        help="Root directory of depth maps")
    parser.add_argument("--output-dir", type=Path,
                        default="data_split/dsec_nbin3",
                        help="Output directory for corrected split files")
    
    args = parser.parse_args()
    
    print(f"Events root: {args.events_root}")
    print(f"Depth root: {args.depth_root}")
    print(f"Output dir: {args.output_dir}")
    
    create_dsec_splits(args.events_root, args.depth_root, args.output_dir)


if __name__ == "__main__":
    main()
