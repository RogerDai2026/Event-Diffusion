#!/usr/bin/env python3
"""
Batch extract events from all DSEC sequences.
This script processes all sequences in the DSEC dataset and extracts events
aligned with depth timestamps.
"""

import argparse
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm


def find_dsec_sequences(dsec_root: Path):
    """Find all DSEC sequences with both events and disparity data."""
    events_root = dsec_root / "events_raw"
    disparity_root = dsec_root / "disparity_maps"
    
    sequences = []
    
    if not events_root.exists() or not disparity_root.exists():
        print(f"Error: Missing directories in {dsec_root}")
        print(f"  events_raw: {events_root.exists()}")
        print(f"  disparity_maps: {disparity_root.exists()}")
        return sequences
    
    # Find all sequence directories
    for seq_dir in events_root.iterdir():
        if not seq_dir.is_dir():
            continue
            
        events_h5 = seq_dir / "events" / "left" / "events.h5"
        disparity_dir = disparity_root / seq_dir.name
        depth_timestamps = disparity_dir / "disparity" / "timestamps.txt"
        
        if events_h5.exists() and depth_timestamps.exists():
            sequences.append({
                'name': seq_dir.name,
                'events_h5': events_h5,
                'depth_timestamps': depth_timestamps,
                'disparity_dir': disparity_dir
            })
        else:
            print(f"Warning: Incomplete sequence {seq_dir.name}")
            print(f"  events.h5: {events_h5.exists()}")
            print(f"  timestamps.txt: {depth_timestamps.exists()}")
    
    return sequences


def extract_sequence_events(seq_info, events_output_dir: Path, overwrite: bool = False):
    """Extract events for a single sequence."""
    seq_name = seq_info['name']
    events_h5 = seq_info['events_h5']
    depth_timestamps = seq_info['depth_timestamps']
    
    # Create output directory
    seq_output_dir = events_output_dir / seq_name
    seq_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed (unless overwrite)
    if not overwrite:
        existing_files = list(seq_output_dir.glob("*.npz"))
        if existing_files:
            print(f"[{seq_name}] Already processed ({len(existing_files)} files), skipping...")
            return True
    
    print(f"[{seq_name}] Processing...")
    
    # Run extraction script
    cmd = [
        sys.executable,
        "src/utils/event/dataset_preprocess/DSEC/generate_events_h5.py",
        "--events-h5", str(events_h5),
        "--depth-timestamps", str(depth_timestamps),
        "--events-out", str(seq_output_dir),
        "--window-ms", "50"
    ]
    
    if overwrite:
        cmd.append("--overwrite")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        if result.returncode != 0:
            print(f"[{seq_name}] ERROR:")
            print(result.stderr)
            return False
        else:
            # Count generated files
            npz_files = list(seq_output_dir.glob("*.npz"))
            print(f"[{seq_name}] SUCCESS: Generated {len(npz_files)} event files")
            return True
    except Exception as e:
        print(f"[{seq_name}] EXCEPTION: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch extract DSEC events")
    parser.add_argument("--dsec-root", type=Path, required=True,
                        help="Root directory of DSEC dataset (contains events_raw/ and disparity_maps/)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for extracted events")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing extracted events")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--sequences", nargs="+", 
                        help="Specific sequences to process (default: all)")
    
    args = parser.parse_args()
    
    # Find sequences
    print(f"Scanning DSEC root: {args.dsec_root}")
    sequences = find_dsec_sequences(args.dsec_root)
    
    if not sequences:
        print("No valid sequences found!")
        return 1
    
    # Filter sequences if specified
    if args.sequences:
        sequences = [s for s in sequences if s['name'] in args.sequences]
        if not sequences:
            print(f"No sequences found matching: {args.sequences}")
            return 1
    
    print(f"Found {len(sequences)} sequences to process:")
    for seq in sequences:
        print(f"  - {seq['name']}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process sequences
    if args.workers == 1:
        # Sequential processing with progress bar
        success_count = 0
        with tqdm(sequences, desc="Processing sequences", unit="seq") as pbar:
            for seq in pbar:
                pbar.set_postfix(current=seq['name'])
                if extract_sequence_events(seq, args.output_dir, args.overwrite):
                    success_count += 1
                pbar.set_postfix(current=seq['name'], success=f"{success_count}/{pbar.n+1}")
    else:
        # Parallel processing with progress bar
        success_count = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Submit all jobs
            future_to_seq = {
                executor.submit(extract_sequence_events, seq, args.output_dir, args.overwrite): seq
                for seq in sequences
            }
            
            # Process completed jobs with progress bar
            with tqdm(total=len(sequences), desc="Processing sequences", unit="seq") as pbar:
                for future in as_completed(future_to_seq):
                    seq = future_to_seq[future]
                    try:
                        if future.result():
                            success_count += 1
                        pbar.set_postfix(
                            current=seq['name'], 
                            success=f"{success_count}/{pbar.n+1}"
                        )
                    except Exception as e:
                        print(f"\n[{seq['name']}] Exception: {e}")
                    finally:
                        pbar.update(1)
    
    print(f"\nCompleted: {success_count}/{len(sequences)} sequences processed successfully")
    
    return 0 if success_count == len(sequences) else 1


if __name__ == "__main__":
    sys.exit(main())
