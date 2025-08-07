# src/models/pipeline/vae_to_unet.py
import torch
from omegaconf import DictConfig
from typing import Any, Dict, Tuple, Optional
import torch.nn.functional as F

from src.utils.event.event_data_utils import resize_to_multiple_of_16
from lightning.pytorch import LightningModule
from src.models.baselines.cnn.vae_module import LatentVAELitModule
from src.models.baselines.cnn.corrdiff_unet import CorrDiffEventLitModule
from src.utils.corrdiff_utils.inference import regression_step_only

class VaeCorrDiffPipelineLitModule(LightningModule):
    def __init__(
        self,
        vae: DictConfig,
        regressor: CorrDiffEventLitModule,
        allow_resize: bool = True,
    ):
        super().__init__()
        # stash these for later
        self.save_hyperparameters("vae", "allow_resize")

        # only wire up the regressor here
        self.regressor = regressor
        self.vae_cfg = vae
        self.allow_resize = allow_resize
        # placeholder for your VAE
        self.vae: Optional[LatentVAELitModule] = None

    def setup(self, stage: Optional[str] = None):

        # 1) build the VAE
        cfg = self.vae_cfg
        self.vae = LatentVAELitModule(
            cfg_path=cfg.cfg_path,
            allow_resize=cfg.allow_resize,
            compile=cfg.compile,
        )

        # 2) optionally JIT‐compile (before loading weights!)
        if cfg.compile and stage == "fit":
            self.vae.autoencoder = torch.compile(self.vae.autoencoder)

        # 3) load its pretrained weights
        ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
        self.vae.load_state_dict(ckpt["state_dict"], strict=False)

        # 4) freeze & switch to eval
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()
        del self.vae.autoencoder.decoder

    def configure_optimizers(self):
        # just train the regressor
        return self.regressor.configure_optimizers()

    def _encode_to_latent(self, x):
        # use the VAE’s encode -> quant_conv
        with torch.no_grad():
            # 1) run encoder → moments
            posterior = self.vae.autoencoder.encode(x)
            # 2) sample the latent (shape [B, 3, H', W'])
            z = posterior.sample()
        # if self.allow_resize:
        #     z = resize_to_multiple_of_16(z)
            z = self._pad_latent(z)
        return z

    def training_step(self, batch, batch_idx):
        cond, gt = self._generate_condition(batch)    # [B,5,720,1280], etc.
        # if self.allow_resize:
        #     cond = resize_to_multiple_of_16(cond)

        # 1) get latent [B,3,180,320]
        z = self._encode_to_latent(cond)

        # 2) forward & loss via your existing UNet criterion
        pred = self.regressor.forward(z)
        gt = F.interpolate(
            gt,
            size=(180, 320),      # (180, 320)
            mode="bilinear",
            align_corners=False,
        )
        gt = self._pad_latent(gt)
        # gt = F.interpolate(
        #     gt,
        #     size=(192, 320),      # (180, 320)
        #     mode="bilinear",
        #     align_corners=False,
        #     antialias=True,
        # )

        loss, _ = self.regressor.criterion(
            net=self.regressor.net,
            img_clean=gt,       # ground truth depth [B,1,180,320]
            img_lr=z,           # our latent “low-res” [B,3,180,320]
        )
        loss = loss.mean()
        self.log("train/loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self._fill_missing_keys(batch)
        cond, gt = self._generate_condition(batch)
        if self.allow_resize:
            cond = resize_to_multiple_of_16(cond)
        z = self._encode_to_latent(cond)
        gt = F.interpolate(
            gt,
            size=(180, 320),      # (180, 320)
            mode="bilinear",
            align_corners=False,
        )
        gt = self._pad_latent(gt)
        # gt = F.interpolate(
        #     gt,
        #     size=(192, 320),      # (180, 320)
        #     mode="bilinear",
        #     align_corners=False,
        #     antialias=True,
        # )
        batch['depth_raw_norm'] = gt
        loss, pred = self.regressor.criterion(
            net=self.regressor.net,
            img_clean=gt,
            img_lr=z,
        )
        loss = loss.mean()
        self.log("val/loss", loss, on_epoch=True, sync_dist=True)
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

    def _fill_missing_keys(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Fill missing keys in the batch dictionary. In this example,
        if 'depth_raw_norm' is not provided, compute it using the
        datamodule’s depth_transform method.
        """
        if 'depth_raw_norm' not in batch_dict:
            # Here we assume that the datamodule has a `depth_transform` callable.
            batch_dict['depth_raw_norm'] = self.trainer.datamodule.depth_transform(batch_dict['depth_raw_linear'])
        return batch_dict

    def _generate_condition(self, batch_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate the condition and target tensors from the batch dictionary.
        This implementation uses a `scaler` which must be provided by subclasses.
        """
        # The scaler method or property is expected to be implemented by the child.
        # condition = self.scaler(batch_dict['rgb_norm'])
        condition = batch_dict['rgb_norm']
        y = batch_dict['depth_raw_norm']
        return condition, y

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
