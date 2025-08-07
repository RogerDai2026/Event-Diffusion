import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningModule, LightningDataModule
from hydra.utils import instantiate
from typing import Optional, Dict

from src.data.depth.cached_latent_dataset import CachedLatentDepthDataset
from src.models.baselines.cnn.corrdiff_unet import CorrDiffEventLitModule
from src.utils.corrdiff_utils.inference import regression_step_only

class LatentRegressorModule(LightningModule):
    """
    Wraps a CorrDiffEventLitModule to train directly on VAE latents.
    Expects a DictConfig `regressor` that instantiates CorrDiffEventLitModule.
    """
    def __init__(self, regressor: Dict):
        super().__init__()
        # save the entire regressor config for checkpointing
        self.save_hyperparameters('regressor')
        # instantiate the CorrDiff UNet lit module
        self.regressor: CorrDiffEventLitModule = instantiate(self.hparams.regressor)

    def configure_optimizers(self):
        return self.regressor.configure_optimizers()

    def forward(self, z, global_index=None):
        return self.regressor.forward(z, global_index)

    def training_step(self, batch, batch_idx):
        z  = batch['rgb_norm']
        gt = batch['depth_raw_norm']
        # z  = self._pad_latent(z)
        # gt = self._pad_latent(gt)

        # forward + loss
        loss, _ = self.regressor.criterion(
            net=self.regressor.net,
            img_clean=gt,
            img_lr=z,
        )
        loss = loss.mean()
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        z  = batch['rgb_norm']
        gt = batch['depth_raw_norm']
        # z  = self._pad_latent(z)
        # gt = self._pad_latent(gt)
        batch['depth_raw_norm'] = gt
        loss, _ = self.regressor.criterion(
            net=self.regressor.net,
            img_clean=gt,
            img_lr=z,
        )
        loss = loss.mean()
        self.log('val/loss', loss, on_epoch=True, on_step=False)
        step_output = {"batch_dict": batch, "condition": z}
        return step_output

    def sample(self, condition: torch.Tensor) -> torch.Tensor:
        # optionally resize here, if you want the same logic as train/val
        B, C, H, W = condition.shape
        latent_shape = (B, 1, H, W)
        # enforce the "regression" time-step t=1.0 and float64, as in the original
        return regression_step_only(
            net=self.regressor.net,
            img_lr=condition,
            latents_shape=latent_shape,
            lead_time_label=None
        )

    def _pad_latent(self, z, target_size=(192, 320)):
        B, C, H, W = z.shape
        tgt_h, tgt_w = target_size
        # compute padding on each side
        pad_h = tgt_h - H  # e.g. 12
        pad_w = tgt_w - W  # e.g. 0
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return F.pad(z, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)