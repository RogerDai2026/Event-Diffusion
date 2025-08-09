#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from hydra import compose, initialize
from src.data.event_datamodule import EventDataModule

def visualize_custom_samples(num_samples=4):
    """Visualize samples from the custom dataset"""
    
    print("=== Visualizing Custom Dataset Samples ===")
    
    # Load configuration
    with initialize(version_base=None, config_path='configs/data', job_name='visualize_custom'):
        config = compose(config_name='event_custom')
    
    # Create and setup data module
    data_module = EventDataModule(
        data_config=config.data_config,
        augmentation_args=config.augmentation_args,
        depth_transform_args=config.depth_transform_args,
        batch_size=1,  # Use batch size 1 for visualization
        num_workers=0,  # No multiprocessing for debugging
        pin_memory=config.pin_memory,
        seed=config.seed
    )
    
    data_module.setup(stage='fit')
    
    # Get datasets
    train_dataset = data_module.train_dataset
    val_dataset = data_module.val_dataset
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Create visualization
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        print(f"\nProcessing sample {i}...")
        
        # Get a sample from train dataset
        sample = train_dataset[i]
        
        print(f"Sample keys: {list(sample.keys())}")
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}, range=[{value.min():.3f}, {value.max():.3f}]")
        
        # Extract data
        rgb_norm = sample['rgb_norm'].numpy()  # [C, H, W] normalized to [-1, 1]
        depth_raw_norm = sample['depth_raw_norm'].numpy()  # [1, H, W] normalized
        depth_raw_linear = sample['depth_raw_linear'].numpy()  # [1, H, W] in meters
        valid_mask = sample['valid_mask_raw'].numpy()  # [1, H, W] boolean mask
        
        # Convert rgb from [C, H, W] to [H, W, C] and from [-1, 1] to [0, 1]
        rgb_display = np.transpose(rgb_norm, (1, 2, 0))  # [H, W, C]
        rgb_display = (rgb_display + 1) / 2  # [-1, 1] -> [0, 1]
        rgb_display = np.clip(rgb_display, 0, 1)
        
        # Convert depth from [1, H, W] to [H, W]
        depth_norm_display = depth_raw_norm[0]  # Remove channel dimension
        depth_linear_display = depth_raw_linear[0]  # Remove channel dimension
        valid_mask_display = valid_mask[0].astype(float)  # Remove channel dimension
        
        # Plot RGB (event representation)
        axes[i, 0].imshow(rgb_display)
        axes[i, 0].set_title(f'Sample {i}: Event RGB\nShape: {rgb_display.shape}')
        axes[i, 0].axis('off')
        
        # Plot normalized depth
        im1 = axes[i, 1].imshow(depth_norm_display, cmap='plasma', vmin=-1, vmax=1)
        axes[i, 1].set_title(f'Depth Normalized\nRange: [{depth_norm_display.min():.2f}, {depth_norm_display.max():.2f}]')
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # Plot linear depth (in meters)
        im2 = axes[i, 2].imshow(depth_linear_display, cmap='plasma')
        axes[i, 2].set_title(f'Depth Linear (meters)\nRange: [{depth_linear_display.min():.1f}, {depth_linear_display.max():.1f}]')
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        # Plot valid mask
        axes[i, 3].imshow(valid_mask_display, cmap='gray', vmin=0, vmax=1)
        axes[i, 3].set_title(f'Valid Mask\nValid pixels: {valid_mask_display.sum():.0f}/{valid_mask_display.size}')
        axes[i, 3].axis('off')
        
        # Print some statistics
        print(f"  Event RGB range: [{rgb_display.min():.3f}, {rgb_display.max():.3f}]")
        print(f"  Depth linear range: [{depth_linear_display.min():.1f}, {depth_linear_display.max():.1f}] meters")
        print(f"  Depth normalized range: [{depth_norm_display.min():.3f}, {depth_norm_display.max():.3f}]")
        print(f"  Valid mask: {valid_mask_display.sum():.0f}/{valid_mask_display.size} pixels valid ({100*valid_mask_display.mean():.1f}%)")
    
    plt.tight_layout()
    
    # Save the visualization
    output_file = 'custom_dataset_samples.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to {output_file}")
    
    # Also show it
    plt.show()

def visualize_batch():
    """Visualize a batch from the dataloader"""
    
    print("\n=== Visualizing Custom Dataset Batch ===")
    
    # Load configuration
    with initialize(version_base=None, config_path='configs/data', job_name='visualize_custom'):
        config = compose(config_name='event_custom')
    
    # Create and setup data module
    data_module = EventDataModule(
        data_config=config.data_config,
        augmentation_args=config.augmentation_args,
        depth_transform_args=config.depth_transform_args,
        batch_size=4,  # Small batch for visualization
        num_workers=0,  # No multiprocessing for debugging
        pin_memory=config.pin_memory,
        seed=config.seed
    )
    
    data_module.setup(stage='fit')
    
    # Get a batch
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"Batch keys: {list(batch.keys())}")
    for key, value in batch.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    batch_size = batch['rgb_norm'].shape[0]
    print(f"\nBatch size: {batch_size}")
    
    # Create visualization for the batch
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 3 * batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Extract data for sample i
        rgb_norm = batch['rgb_norm'][i].numpy()  # [C, H, W]
        depth_raw_linear = batch['depth_raw_linear'][i].numpy()  # [1, H, W]
        valid_mask = batch['valid_mask_raw'][i].numpy()  # [1, H, W]
        
        # Convert for display
        rgb_display = np.transpose(rgb_norm, (1, 2, 0))  # [H, W, C]
        rgb_display = (rgb_display + 1) / 2  # [-1, 1] -> [0, 1]
        rgb_display = np.clip(rgb_display, 0, 1)
        
        depth_display = depth_raw_linear[0]  # Remove channel dimension
        valid_mask_display = valid_mask[0].astype(float)
        
        # Plot
        axes[i, 0].imshow(rgb_display)
        axes[i, 0].set_title(f'Batch {i}: Event RGB')
        axes[i, 0].axis('off')
        
        im = axes[i, 1].imshow(depth_display, cmap='plasma')
        axes[i, 1].set_title(f'Depth (meters)')
        axes[i, 1].axis('off')
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        axes[i, 2].imshow(valid_mask_display, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Valid Mask')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    # Save the batch visualization
    output_file = 'custom_dataset_batch.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Batch visualization saved to {output_file}")
    
    plt.show()

if __name__ == "__main__":
    # Visualize individual samples
    visualize_custom_samples(num_samples=3)
    
    # Visualize a batch
    visualize_batch() 