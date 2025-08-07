from typing import Any, Dict, Tuple, Optional, List
import os, sys

import time
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from omegaconf import OmegaConf
import torch.nn.functional as F

from src.utils.latent_diffusion.ldm.util import instantiate_from_config
from src.utils.latent_diffusion.ldm.models.autoencoder import AutoencoderKL
from src.utils.latent_diffusion.ldm.modules.losses.nll_kl_loss import NLLKLLoss, WeightedNLLKLLoss, EventL1NLLKLLoss # official latent_diffusion autoencoder
from src.utils.latent_diffusion.ldm.modules.losses.contperceptual import LPIPSWithDiscriminator

from src.models.event.generic_e2d_module import GenericE2DModule
from src.utils.event.event_data_utils import resize_to_multiple_of_16
from src.utils.helper import yprint


class LatentVAELitModule(GenericE2DModule):
    """
    LightningModule wrapping CompVis’s AutoencoderKL trained
    with a pure NLL + KL objective (no perceptual or GAN terms),
    fully channel-agnostic on 5-bin event tensors.
    Prints mean and std of validation losses each epoch.
    """

    def __init__(
            self,
            cfg_path: str,
            lr: float = 0.0001,
            beta_kl: float = 1.0,
            allow_resize: bool = False,
            compile: bool = False,
            checkpoint_path : str=None,
            nll_weight : float = 2.0,
    ) -> None:
        super().__init__()
        # Save hyperparameters for logging/checkpointing
        self.save_hyperparameters("cfg_path", "lr", "beta_kl", "allow_resize", "compile","checkpoint_path", "nll_weight")

        # Load VAE config and instantiate AutoencoderKL
        self.cfg = OmegaConf.load(cfg_path)
        self.autoencoder: AutoencoderKL = instantiate_from_config(self.cfg.model)
        # del self.autoencoder.loss

        # Use NLL+KL loss (learned logvar + KL term)
        self.loss_fn = WeightedNLLKLLoss(logvar_init=0.0, kl_weight=self.hparams.beta_kl, event_weight=self.hparams.nll_weight)
        # #, event_weight=self.hparams.nll_weight
        # self.loss_fn = EventL1NLLKLLoss(logvar_init=0.0, kl_weight=self.hparams.beta_kl)

        ## For freeze decoder and check encoder in latent
        if self.hparams.checkpoint_path:
            ckpt = torch.load(self.hparams.checkpoint_path, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
                    # filter only decoder.* keys
            decoder_state = {
                        k.replace("decoder.", ""): v
                        for k, v in state.items()
                        if k.startswith("decoder.")
            }
             # load into decoder
            self.autoencoder.decoder.load_state_dict(decoder_state, strict=True)
                    # freeze decoder params
            for p in self.autoencoder.decoder.parameters():
                p.requires_grad = False
            print(f">> Loaded & frozen decoder from {self.hparams.checkpoint_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        # Optionally compile the autoencoder for speed
        if self.hparams.compile and stage == "fit":
            self.autoencoder = torch.compile(self.autoencoder)
            print("AutoencoderKL compiled with torch.compile.")

    def configure_optimizers(self) -> Dict[str, Any]:
        # Single optimizer over encoder, decoder, quantizers, and loss logvar
        params = (
                list(self.autoencoder.encoder.parameters())
                + list(self.autoencoder.decoder.parameters())
                + list(self.autoencoder.quant_conv.parameters())
                + list(self.autoencoder.post_quant_conv.parameters())
                + [self.loss_fn.logvar]
        )
        optimizer = torch.optim.Adam(params, lr=self.hparams.lr)
        return {"optimizer": optimizer}

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        # Forward pass through AutoencoderKL: returns (recon, posterior)
        return self.autoencoder(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # Extract your 5-bin event tensor
        x, _ = self._generate_condition(batch)  # [B, 5, H, W]
        if self.hparams.allow_resize:
            x = resize_to_multiple_of_16(x)

        # Reconstruct and get posterior
        recon, posterior = self(x)

        warmup_epochs = 1 #no warmup
        α = min(1.0, (self.current_epoch + 1) / warmup_epochs)

        loss, log_dict = self.loss_fn(
            inputs=x,
            reconstructions=recon,
            posteriors=posterior,
            split="train",
            kl_weight=α * 1.0e-06 #currently hard coded
        )

        # # Log losses
        self.log("train/total_loss", log_dict["train/total_loss"], on_step=True, on_epoch=False)
        self.log("train/nll_loss", log_dict["train/nll_loss"], on_step=True, on_epoch=False)
        self.log("train/kl_loss", log_dict["train/kl_loss"], on_step=True, on_epoch=False)
        self.log("train/logvar", log_dict["train/logvar"].mean(), on_step=True, on_epoch=False)

        # self.log("train/l1_loss", log_dict["train/l1_loss"],on_step=True, on_epoch=False)
        # self.log("train/kl_loss", log_dict["train/kl_loss"],  on_step=True, on_epoch=False)
        # self.log("train/total_loss", log_dict["train/total_loss"], on_step=True, on_epoch=False)
        # self.log("train/nll_loss", log_dict["train/nll_loss"], on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        batch = self._fill_missing_keys(batch)
        x, _ = self._generate_condition(batch)
        if self.hparams.allow_resize:
            x = resize_to_multiple_of_16(x)

        recon, posterior = self(x)

        loss, log_dict = self.loss_fn(
            inputs=x,
            reconstructions=recon,
            posteriors=posterior,
            split="val",
            kl_weight=1.0e-06
        )
        with torch.no_grad():
            mse = F.mse_loss(recon, x).item()
            mae = F.l1_loss(recon, x).item()
            # dynamic range for PSNR
            dr = (x.max() - x.min()).clamp_min(1e-12)
            psnr = (10 * torch.log10(dr ** 2 / (mse + 1e-12))).item()

        # log scalars once for the whole epoch
        self.log("val/mse", mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mae", mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/psnr", psnr, on_step=False, on_epoch=True, sync_dist=True)
        # Log per-batch validation loss
        self.log("val/total_loss", log_dict["val/total_loss"], on_step=False, on_epoch=True, sync_dist=True)
        return {"val_loss": log_dict["val/total_loss"], "input": x}

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        batch = self._fill_missing_keys(batch)
        x, _ = self._generate_condition(batch)
        if self.hparams.allow_resize:
            x = resize_to_multiple_of_16(x)

        start = time.time()
        recon, posterior = self(x)
        elapsed = time.time() - start
        per_image = elapsed / x.size(0)
        print(f"[Test] batch {batch_idx}: total recon time {elapsed:.4f}s, per image {per_image:.4f}s")

        self.log("test/recon_time_batch", elapsed, prog_bar=False, on_step=False, on_epoch=True,  sync_dist=True)
        # (optional) log a scalar test loss
        self.log("test/mse", F.mse_loss(recon, x),
                 on_step=False, on_epoch=True, sync_dist=True)

        # return inputs so that VAELogger.log_samples will visualize them
        return {"input": x, "recon": recon}

    def sample(self, condition: torch.Tensor) -> torch.Tensor:
        # Generate reconstructions for VAELogger visualization
        if self.hparams.allow_resize:
            condition = resize_to_multiple_of_16(condition)
        recon, _ = self(condition)
        return recon



#
#
#
#
# # this is used for lpips module with freeze decoder
# """
# Latent VAE LightningModule with LPIPS + GAN discriminator + NLL + KL loss.
# """
# import os
# import time
# from typing import Any, Dict, Optional, Tuple, List
#
# import torch
# import torch.nn.functional as F
# from lightning.pytorch import LightningModule
# from torch import nn
# from omegaconf import OmegaConf
#
# from src.utils.latent_diffusion.ldm.util import instantiate_from_config
# from src.utils.latent_diffusion.ldm.models.autoencoder import AutoencoderKL
# from src.utils.latent_diffusion.ldm.modules.losses.contperceptual import LPIPSWithDiscriminator
# from src.models.event.generic_e2d_module import GenericE2DModule
# from src.utils.event.event_data_utils import resize_to_multiple_of_16
# from src.utils.helper import yprint
#
#
# class LatentVAELitModule(GenericE2DModule):
#     """
#     LightningModule wrapping a VAE (AutoencoderKL) trained with a GAN + NLL + KL loss.
#     The LPIPS term is disabled (perceptual_weight=0.0).
#     """
#     def __init__(
#         self,
#         cfg_path: str,
#         lr: float = 1e-4,
#         allow_resize: bool = False,
#         compile: bool = False,
#         checkpoint_path: Optional[str] = None,
#     ) -> None:
#         super().__init__()
#         # Save hyperparameters
#         self.save_hyperparameters("cfg_path", "lr", "allow_resize", "compile", "checkpoint_path")
#         self.automatic_optimization = False
#
#         # 1) Load VAE config and instantiate AutoencoderKL
#         self.cfg = OmegaConf.load(cfg_path)
#         self.autoencoder: AutoencoderKL = instantiate_from_config(self.cfg.model)
#
#         # For freeze decoder and check encoder in latent
#         if self.hparams.checkpoint_path:
#             ckpt = torch.load(self.hparams.checkpoint_path, map_location="cpu", weights_only=False)
#             state = ckpt.get("state_dict", ckpt)
#                             # filter only decoder.* keys
#             decoder_state = {
#                         k.replace("decoder.", ""): v
#                         for k, v in state.items()
#                         if k.startswith("decoder.")
#             }
#                      # load into decoder
#             self.autoencoder.decoder.load_state_dict(decoder_state, strict=True)
#                             # freeze decoder params
#             for p in self.autoencoder.decoder.parameters():
#                 p.requires_grad = False
#             print(f">> Loaded & frozen decoder from {self.hparams.checkpoint_path}")
#
#         # 3) Define combined loss: GAN + NLL + KL, no LPIPS
#         self.loss_fn = LPIPSWithDiscriminator(
#             disc_start=self.cfg.model.params.lossconfig.params.disc_start,
#             logvar_init=0.0,
#             kl_weight=self.cfg.model.params.lossconfig.params.kl_weight,
#             pixelloss_weight=1.0,
#             disc_num_layers=3,
#             disc_in_channels=self.cfg.model.params.ddconfig.in_channels,
#             disc_factor=1.0,
#             disc_weight=1.0,
#             perceptual_weight=0.0,
#         )
#
#     def setup(self, stage: Optional[str] = None) -> None:
#         # Optionally compile model for speed
#         if self.hparams.compile and stage == "fit":
#             self.autoencoder = torch.compile(self.autoencoder)
#             yprint("Compiled Autoencoder with torch.compile()")
#
#     def configure_optimizers(self) -> List[torch.optim.Optimizer]:
#         gen_params = (
#                 list(self.autoencoder.encoder.parameters()) +
#                 list(self.autoencoder.quant_conv.parameters()) +
#                 list(self.autoencoder.post_quant_conv.parameters()) +
#                 [self.loss_fn.logvar]
#         )
#         opt_g = torch.optim.Adam(gen_params, lr=self.hparams.lr)
#
#         # 2) discriminator
#         disc_params = list(self.loss_fn.discriminator.parameters())
#         opt_d = torch.optim.Adam(disc_params, lr=self.hparams.lr)
#
#         return [opt_g, opt_d]
#
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
#         # returns (reconstruction, posterior)
#         return self.autoencoder(x)
#
#     def training_step(self, batch: Any, batch_idx: int) -> None:
#         # get both optimizers
#         opt_g, opt_d = self.optimizers()
#
#         # prepare input
#         x, _ = self._generate_condition(batch)
#         if self.hparams.allow_resize:
#             x = resize_to_multiple_of_16(x)
#
#         last_layer = None
#         if hasattr(self.autoencoder.decoder, "conv_out"):
#             last_layer = self.autoencoder.decoder.conv_out.weight  # or whatever parameter you want to use
#
#         # forward
#         recon, posterior = self(x)
#
#         # ----- generator update -----
#         # compute generator loss (NLL+KL+GAN) via optimizer_idx=0
#         g_loss, g_logs = self.loss_fn(
#             inputs=x,
#             reconstructions=recon,
#             posteriors=posterior,
#             optimizer_idx=0,
#             global_step=self.global_step,
#             split="train",
#             last_layer=last_layer,  # pass it in so fallback isn't used
#         )
#         # backward + step generator
#         self.manual_backward(g_loss)
#         opt_g.step()
#         opt_g.zero_grad()
#
#         # log generator metrics
#         self.log("train/total_loss", g_logs["train/total_loss"], on_step=True, prog_bar=True)
#         self.log("train/nll_loss", g_logs["train/nll_loss"], on_step=True)
#         self.log("train/kl_loss", g_logs["train/kl_loss"], on_step=True)
#
#         # ----- discriminator update -----
#         # compute discriminator loss via optimizer_idx=1
#         d_loss, d_logs = self.loss_fn(
#             inputs=x,
#             reconstructions=recon,
#             posteriors=posterior,
#             optimizer_idx=1,
#             global_step=self.global_step,
#             split="train",
#             last_layer=last_layer,
#         )
#         # backward + step discriminator
#         self.manual_backward(d_loss)
#         opt_d.step()
#         opt_d.zero_grad()
#
#         # (optional) log discriminator metrics
#         self.log("train/disc_loss", d_logs["train/disc_loss"], on_step=True)
#
#         return
#
#     @torch.no_grad()
#     def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
#         # 1) Prepare input just like in training
#         batch = self._fill_missing_keys(batch)
#         x, _ = self._generate_condition(batch)       # [B,5,H,W]
#         if self.hparams.allow_resize:
#             x = resize_to_multiple_of_16(x)
#
#         # 2) Forward pass
#         recon, posterior = self(x)
#
#         # 3) Compute reconstruction loss (NLL) with learned logvar:
#         #    scaled L1 + logvar term
#         rec_term = torch.abs(x - recon) / torch.exp(self.loss_fn.logvar) \
#                    + self.loss_fn.logvar
#         nll_loss = rec_term.mean()
#
#         # 4) Compute KL loss
#         kl_term = posterior.kl()
#         kl_loss = kl_term.mean()
#
#         # 5) Total
#         total_loss = nll_loss + self.loss_fn.kl_weight * kl_loss
#
#         # 6) Log scalars (accumulate over epoch)
#         self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True)
#         self.log("val/nll_loss",     nll_loss,     on_step=False, on_epoch=True, sync_dist=True)
#         self.log("val/kl_loss",      kl_loss,      on_step=False, on_epoch=True, sync_dist=True)
#
#         # 7) Return for any epoch‐end hooks
#         return {"val_loss": total_loss , "input": x}
#
#
#     def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
#         x, _ = self._generate_condition(batch)
#         if self.hparams.allow_resize:
#             x = resize_to_multiple_of_16(x)
#
#         start = time.time()
#         recon, _ = self(x)
#         elapsed = time.time() - start
#         self.log("test/inference_time", elapsed, on_epoch=True)
#         return {"reconstruction": recon}
#
#     def sample(self, condition: torch.Tensor) -> torch.Tensor:
#         if self.hparams.allow_resize:
#             condition = resize_to_multiple_of_16(condition)
#         recon, _ = self(condition)
#         return recon
#
#
