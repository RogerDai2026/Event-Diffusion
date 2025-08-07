import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningModule, LightningDataModule
from hydra.utils import instantiate
from typing import Optional, Dict
from src.utils.event.event_data_utils import resize_to_multiple_of_16
from src.data.depth.cached_latent_dataset import CachedLatentDepthDataset
from src.models.baselines.cnn.corrdiff_unet import CorrDiffEventLitModule
from src.utils.event.depth_transform import DepthNormalizerBase, get_depth_normalizer
from typing import Union
from torch.utils.data._utils.collate import default_collate




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
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        seed: int, depth_transform_args: dict, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.depth_transform = None


    def setup(self, stage: Optional[str] = None):
        self.depth_transform  = get_depth_normalizer(cfg_normalizer=self.hparams.depth_transform_args)
        self.train_dataset = CachedLatentDepthDataset(
            root_dir=self.hparams.root_dir, split="train", depth_transform= self.depth_transform
        )
        self.val_dataset = CachedLatentDepthDataset(
            root_dir=self.hparams.root_dir, split="val",  depth_transform= self.depth_transform
        )

    @staticmethod
    def _resize_collate(batch):
        """
        Resize each tensor in the batch to a multiple of 16, then default_collate.
        """
        for sample in batch:
            # add dummy batch-dim for resize fn, then squeeze it out
            sample['rgb_norm'] = resize_to_multiple_of_16(
                sample['rgb_norm'].unsqueeze(0)
            ).squeeze(0)
            sample['depth_raw_norm'] = resize_to_multiple_of_16(
                sample['depth_raw_norm'].unsqueeze(0)
            ).squeeze(0)
        return default_collate(batch)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self._resize_collate,  # << apply resize before stacking
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self._resize_collate,  # << apply resize before stacking
        )

if __name__ == '__main__':
    # quick test of the DataModule
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/shared/qd8/vae_latent_nbin3')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--depth_transform_args', default = None)
    args = parser.parse_args()

    dm = LatentDataModule(
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory= True,
        depth_transform_args = args.depth_transform_args,
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
