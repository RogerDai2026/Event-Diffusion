#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import argparse
from typing import Any, Dict, Optional, Tuple
# import numpy as np
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, SubsetRandomSampler
from hydra import compose, initialize
from tqdm import tqdm
from src.data.depth import get_dataset, DatasetMode
from src.utils.event.depth_transform import DepthNormalizerBase, get_depth_normalizer

def visualize_down_up(val_loader: DataLoader,
                      down_factor: int = 4,
                      crop_size: int = 256,
                      num_batches: int = 1,
                      device: str = "cpu"):
    """
    For each batch in val_loader:
      1) center-crop to crop_size×crop_size
      2) downsample by down_factor (bilinear)
      3) upsample back to crop_size×crop_size (bilinear)
      4) plot original vs down-up for the first few examples
    """
    for batch_idx, batch in enumerate(val_loader):
        # assume batch is (input_tensor, labels or _) with input_tensor [B,C,H,W]
        batch_dict = batch  # no unpacking
        x = batch_dict["rgb_norm"]
        x = x.to(device)
        B, C, H, W = x.shape

        # 1) center crop
        top  = (H - crop_size) // 2
        left = (W - crop_size) // 2
        x_crop = x[:, :, top:top+crop_size, left:left+crop_size]

        # 2) downsample + 3) upsample
        x_down = F.interpolate(x_crop,
                               scale_factor=1/down_factor,
                               mode="bilinear",
                               align_corners=False,antialias=True)
        x_up   = F.interpolate(x_down,
                               size=(crop_size, crop_size),
                               mode="bilinear",
                               align_corners=False,antialias=True)

        # 4) plot the first min(4,B) examples
        n = min(8, B)
        orig = x_crop[:n]
        recon = x_up[:n]

        # for display we take first 3 channels as RGB
        grid_orig = make_grid(orig[:, :3].cpu(), nrow=n, normalize=True).permute(1,2,0)
        grid_up   = make_grid(recon[:, :3].cpu(), nrow=n, normalize=True).permute(1,2,0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(n*3, 6))
        ax1.imshow(grid_orig.numpy())
        ax1.set_title("Original 256×256 center-crop")
        ax1.axis("off")
        ax2.imshow(grid_up.numpy())
        ax2.set_title(f"Down×{down_factor} → Up back to 256")
        ax2.axis("off")
        plt.tight_layout()
        plt.show()

        if batch_idx + 1 >= num_batches:
            break

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size",    type=int,   default=8)
    p.add_argument("--down-factor",   type=int,   default=4)
    p.add_argument("--crop-size",     type=int,   default=256)
    p.add_argument("--num-batches",   type=int,   default=1)
    p.add_argument("--device",        type=str,   default="cpu")
    args = p.parse_args()

    # --- replace this block with however you build your validation loader ---
    from src.data.event_datamodule import EventDataModule
    with initialize(version_base=None, config_path="../../../configs/data", job_name="evaluation"):
        config = compose(config_name="event_custom_nbin5")
    # config.batch_size = 1
    dm = EventDataModule(data_config=config.data_config, augmentation_args=config.augmentation_args,
                                  depth_transform_args=config.depth_transform_args, batch_size=config.batch_size,
                                  num_workers=config.num_workers, pin_memory=config.pin_memory, seed=config.seed)     # ← pass in your config / paths
    dm.setup("fit")
    val_loader = dm.val_dataloader()
    # ---------------------------------------------------------------

    visualize_down_up(val_loader,
                      down_factor = args.down_factor,
                      crop_size   = args.crop_size,
                      num_batches = args.num_batches,
                      device      = args.device)
