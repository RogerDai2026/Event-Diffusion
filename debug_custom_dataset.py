#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from hydra import compose, initialize
from src.data.event_datamodule import EventDataModule

def debug_dataset_loading():
    """Debug the dataset loading process step by step using EventDataModule"""
    
    print("=== DEBUG: Custom Dataset Loading via EventDataModule ===")
    
    # 1. Check config loading
    print("\n1. Loading configuration...")
    try:
        with initialize(version_base=None, config_path='configs/data', job_name='debug_custom'):
            config = compose(config_name='event_custom')
        print(f"✅ Config loaded successfully!")
        print(f"   Base dir: {config.data_config.base_dir}")
        print(f"   Train name: {config.data_config.train.name}")
        print(f"   Train dir: '{config.data_config.train.dir}'")
        print(f"   Train filenames: {config.data_config.train.filenames}")
        print(f"   io_args: {config.data_config.io_args}")
        print(f"   depth_transform_args: {config.depth_transform_args}")
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False
    
    # 2. Check file paths
    print("\n2. Checking file paths...")
    train_file = config.data_config.train.filenames
    if not train_file.startswith('/'):
        # Relative path - should be relative to project directory
        full_train_path = os.path.join(os.getcwd(), train_file)
    else:
        full_train_path = train_file
    
    print(f"   Train file path: {train_file}")
    print(f"   Full train path: {full_train_path}")
    print(f"   Train file exists: {os.path.exists(full_train_path)}")
    
    if os.path.exists(full_train_path):
        with open(full_train_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        print(f"   Train file has {len(lines)} entries")
        if lines:
            print(f"   Sample entry: {lines[0]}")
    
    # 3. Create EventDataModule (which handles depth_transform properly)
    print("\n3. Creating EventDataModule...")
    try:
        data_module = EventDataModule(
            data_config=config.data_config,
            augmentation_args=config.augmentation_args,
            depth_transform_args=config.depth_transform_args,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            seed=config.seed
        )
        print(f"✅ EventDataModule created successfully!")
    except Exception as e:
        print(f"❌ EventDataModule creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Setup the data module (this creates datasets with proper depth_transform)
    print("\n4. Setting up data module...")
    try:
        data_module.setup(stage='fit')
        print(f"✅ Data module setup successful!")
        print(f"   Train dataset: {type(data_module.train_dataset)}")
        print(f"   Train dataset size: {len(data_module.train_dataset)}")
        print(f"   Val dataset: {type(data_module.val_dataset)}")
        print(f"   Val dataset size: {len(data_module.val_dataset)}")
        print(f"   Depth transform: {type(data_module.depth_transform)}")
    except Exception as e:
        print(f"❌ Data module setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Test loading a sample from the train dataset
    print("\n5. Testing sample loading from train dataset...")
    try:
        train_dataset = data_module.train_dataset
        print(f"   Dataset depth_transform: {train_dataset.depth_transform}")
        print(f"   Dataset depth_transform type: {type(train_dataset.depth_transform)}")
        
        sample = train_dataset[0]
        print(f"✅ Sample loaded successfully!")
        print(f"   Sample keys: {list(sample.keys())}")
        
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"   {key}: {type(value)} = {value}")
                
    except Exception as e:
        print(f"❌ Sample loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. Test creating a dataloader
    print("\n6. Testing dataloader creation...")
    try:
        train_loader = data_module.train_dataloader()
        print(f"✅ Train dataloader created successfully!")
        print(f"   Number of batches: {len(train_loader)}")
        
        # Try to get one batch
        batch = next(iter(train_loader))
        print(f"✅ Batch loaded successfully!")
        print(f"   Batch keys: {list(batch.keys())}")
        
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
                
    except Exception as e:
        print(f"❌ Dataloader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All debug checks passed!")
    return True

if __name__ == "__main__":
    debug_dataset_loading() 