#!/usr/bin/env python3

import torch
import psutil
import os
import sys
import hydra
from omegaconf import DictConfig
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.event_datamodule import EventDataModule

def get_gpu_memory():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0

def get_cpu_memory():
    """Get current CPU memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3  # GB

def test_config(config_name, max_batches=3):
    """Test memory usage for a specific config"""
    print(f"\n{'='*50}")
    print(f"Testing config: {config_name}")
    print(f"{'='*50}")
    
    try:
        # Load config
        with hydra.initialize(version_base="1.3", config_path="configs"):
            cfg = hydra.compose(config_name="train", overrides=[f"data={config_name}"])
        
        # Create datamodule
        datamodule = EventDataModule(**cfg.data)
        datamodule.setup()
        
        # Get train dataloader
        train_loader = datamodule.train_dataloader()
        
        print(f"Batch size: {cfg.data.batch_size}")
        print(f"Image size: {cfg.data.data_config.train.resize_to_hw}")
        print(f"Channels: {cfg.data.data_config.num_input_chs}")
        
        # Test loading batches
        initial_gpu = get_gpu_memory()
        initial_cpu = get_cpu_memory()
        
        print(f"Initial GPU memory: {initial_gpu:.2f} GB")
        print(f"Initial CPU memory: {initial_cpu:.2f} GB")
        
        max_gpu = initial_gpu
        max_cpu = initial_cpu
        
        for i, batch in enumerate(train_loader):
            if i >= max_batches:
                break
                
            # Move to GPU if available
            if torch.cuda.is_available():
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            current_gpu = get_gpu_memory()
            current_cpu = get_cpu_memory()
            
            max_gpu = max(max_gpu, current_gpu)
            max_cpu = max(max_cpu, current_cpu)
            
            print(f"Batch {i+1}: GPU {current_gpu:.2f} GB, CPU {current_cpu:.2f} GB")
            
            # Print tensor shapes
            if i == 0:
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.shape} ({value.dtype})")
        
        print(f"Peak GPU memory: {max_gpu:.2f} GB")
        print(f"Peak CPU memory: {max_cpu:.2f} GB")
        print(f"GPU memory increase: {max_gpu - initial_gpu:.2f} GB")
        print(f"CPU memory increase: {max_cpu - initial_cpu:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with {config_name}: {e}")
        return False
    finally:
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    print("Testing memory usage for different custom dataset configurations")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test different configurations
    configs_to_test = [
        "event_custom_small",   # 256x344
        "event_custom_medium",  # 384x512  
        "event_custom",         # 480x640
    ]
    
    results = {}
    for config in configs_to_test:
        success = test_config(config)
        results[config] = success
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for config, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{config}: {status}")
    
    print(f"\nRecommendation:")
    print(f"- Start with event_custom_small (256x344) - same as CARLA")
    print(f"- If that works, try event_custom_medium (384x512)")
    print(f"- Only use event_custom (480x640) if you have plenty of GPU memory")

if __name__ == "__main__":
    main() 