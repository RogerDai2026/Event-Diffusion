import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningModule, LightningDataModule
from hydra.utils import instantiate
from typing import Optional, Dict

from src.data.depth.cached_latent_dataset import CachedLatentDepthDataset
from src.models.baselines.cnn.corrdiff_unet import CorrDiffEventLitModule


class LatentDataModule(LightningDataModule):
    """
    LightningDataModule for precomputed VAE latents and depth tensors.
    Expects on-disk structure:
      root_dir/
        train/latents/*.pt
        train/depths/*.pt
        val/latents/*.pt
        val/depths/*.pt
    """
    def __init__(
        self,
        root_dir: str,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = CachedLatentDepthDataset(
            root_dir=self.hparams.root_dir, split="train"
        )
        self.val_dataset = CachedLatentDepthDataset(
            root_dir=self.hparams.root_dir, split="val"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

if __name__ == '__main__':
    # quick test of the DataModule
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/shared/qd8/vae_latents')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    dm = LatentDataModule(
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        seed=42,
    )
    dm.setup()

    train_loader = dm.train_dataloader()
    print(f"Train batches: {len(train_loader)}")
    for i, batch in enumerate(train_loader):
        print(f"Train Batch {i}: rgb_norm {batch['rgb_norm'].shape}, depth_raw_norm {batch['depth_raw_norm'].shape}")
        if i >= 2:
            break

    val_loader = dm.val_dataloader()
    print(f"Val batches: {len(val_loader)}")
    for i, batch in enumerate(val_loader):
        print(f"Val Batch {i}: rgb_norm {batch['rgb_norm'].shape}, depth_raw_norm {batch['depth_raw_norm'].shape}")
        if i >= 2:
            break
