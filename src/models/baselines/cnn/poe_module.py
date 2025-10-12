# src/models/multimodal/poe_ae_module.py
from typing import Any, Dict, Tuple, Optional, List
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning.pytorch import LightningModule
from omegaconf import OmegaConf

from src.utils.event.alignment_loss import w2_diag_gaussians, swd_from_params
from src.utils.latent_diffusion.ldm.util import instantiate_from_config
from src.utils.latent_diffusion.ldm.models.autoencoder import AutoencoderKL
from src.utils.latent_diffusion.ldm.modules.losses.nll_kl_loss import WeightedNLLKLLoss, WeightedNLLKLLossBalanced
from src.utils.latent_diffusion.ldm.modules.losses.contperceptual import LPIPSWithDiscriminator
from src.models.event.generic_e2d_module import GenericE2DModule
from src.utils.event.event_data_utils import resize_to_multiple_of_16


class PoEPosterior:
    """
    Minimal wrapper that mimics the posterior used by CompVis AutoencoderKL
    (i.e., has .kl()) so it can be passed into your losses.
    Stores diagonal Gaussian parameters (mean/logvar) for q(z|X).
    Shapes may be [B, C, H, W] – ops are elementwise.
    """
    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor):
        self.mean = mean
        self.logvar = logvar

    def sample(self) -> torch.Tensor:
        std = torch.exp(0.5 * self.logvar)
        return self.mean + torch.randn_like(std) * std

    def kl(self) -> torch.Tensor:
        # KL[N(mu, var) || N(0, I)] (diag). Returns tensor with same trailing dims as mean/logvar.
        return -0.5 * (1.0 + self.logvar - self.mean.pow(2) - self.logvar.exp())



def poe_fuse(mus: List[torch.Tensor], logvars: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Product-of-experts for diagonal Gaussians.
    T = sum_i (1/var_i); mu = (sum_i mu_i/var_i) / T; logvar = log(1/T)
    Elementwise over any trailing dims ([B, C, H, W] works).
    """
    vars_ = [lv.exp() for lv in logvars]
    precisions = [1.0 / v for v in vars_]
    T = torch.stack(precisions, dim=0).sum(dim=0)                            # [B, ...]
    muT = torch.stack([m * p for m, p in zip(mus, precisions)], dim=0).sum(0)
    mu = muT / T
    logvar = torch.log(1.0 / T)
    return mu, logvar


class PoEVAELitModule(GenericE2DModule):
    """
    Product-of-Experts (PoE) autoencoder wrapper that combines the event VAE
    and the depth VAE into a single shared latent space.

    Implements the MVAE training recipe:
      - Joint term uses PoE(q_e, q_d, (optional) prior) with recon losses only,
        then adds ONE explicit KL on the joint posterior.
      - Unimodal auxiliary terms decode from each modality's own posterior and
        include KL_uni inside those terms.
    """

    def __init__(
        self,
        event_cfg_path: str,
        depth_cfg_path: str,
        # Optional checkpoints for full initialization (enc+dec+quant/post-quant)
        event_checkpoint_path: Optional[str] = None,
        depth_checkpoint_path: Optional[str] = None,

        lr: float = 1e-4,
        allow_resize: bool = True,
        compile: bool = False,

        # KL scaling
        beta_kl_joint: float = 1e-6,
        beta_kl_uni: float = 1e-6,

        # PoE knobs
        use_prior_expert: bool = True,
        omega_precision: float = 1.0,
        ##TODO: check whether this shrinkage is optimum?
        omega_event: float = 1.0,
        omega_depth: float = 0.01,

        include_unimodal_terms: bool = True,

        # Event loss specifics
        event_nll_weight: float = 1.5,    # upweight active event voxels
        ##TODO: event_thresh may not be 0.5 after the nbin encoding by e2d(they have bin accumulation)
        event_thresh: float = 0.5,

        # # wassertain distance
        # lambda_w2: float = 0.1,
        # lambda_swd: float = 0.0,

        # Freezing options for staged alignment
        freeze_event_decoder: bool = False,
        freeze_depth_decoder: bool = False,
        freeze_event_encoder: bool = False,
        freeze_depth_encoder: bool = False,

        #weight for stop depth bullying in the start
        lambda_joint_event: float = 1.0,
        lambda_uni_event: float = 1.0,
        ##TODO: Check whether this scaling of loss is correct?
        lambda_joint_depth: float = 1.0,  # << half weight for depth (joint)
        lambda_uni_depth: float = 1.0,  # << half weight for depth (unimodal)

        # Depth image channels
        depth_in_channels: int = 3,  # because KL-f4 depth VAE is RGB
        depth_encoder_expects_rgb: bool = True,
        depth_loss_rgb: bool = False,  # loss already gets 3ch; no loss-only replication needed
        depth_pix_loss_scale: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Instantiate both AEs
        self.event_cfg = OmegaConf.load(event_cfg_path)
        self.depth_cfg = OmegaConf.load(depth_cfg_path)
        self.event_ae: AutoencoderKL = instantiate_from_config(self.event_cfg.model)
        self.depth_ae: AutoencoderKL = instantiate_from_config(self.depth_cfg.model)
        self.depth_in_channels = depth_in_channels

        # Load FULL weights if provided
        if event_checkpoint_path:
            self._load_full_autoencoder(self.event_ae, event_checkpoint_path, strict=False)
        if depth_checkpoint_path:
            self._load_full_autoencoder(self.depth_ae, depth_checkpoint_path, strict=False)
        if self.hparams.depth_pix_loss_scale is None:
            self.hparams.depth_pix_loss_scale = 1.0

        # Optional freezing
        self._maybe_freeze(self.event_ae.decoder, self.hparams.freeze_event_decoder)
        self._maybe_freeze(self.depth_ae.decoder, self.hparams.freeze_depth_decoder)
        self._maybe_freeze(self.event_ae.encoder, self.hparams.freeze_event_encoder)
        self._maybe_freeze(self.depth_ae.encoder, self.hparams.freeze_depth_encoder)

        # Latent dims must match
        assert getattr(self.event_ae, "embed_dim", None) == getattr(self.depth_ae, "embed_dim", None), \
            "event_ae.embed_dim must equal depth_ae.embed_dim for shared latent space."

        # Per-modality losses
        self.event_loss = WeightedNLLKLLoss(
            logvar_init=0.0,
            kl_weight=self.hparams.beta_kl_joint,   # will be overridden to 0.0 in joint branch
            event_weight=self.hparams.event_nll_weight,
            event_thresh=self.hparams.event_thresh
        )
        ##TODO: Check whether 1.5k warmup is needed?
        # self.event_loss = WeightedNLLKLLossBalanced(
        #     logvar_init=0.0,
        #     kl_weight=self.hparams.beta_kl_joint,  # overridden to 0.0 in joint branch
        #     event_weight=self.hparams.event_nll_weight,
        #     event_thresh=self.hparams.event_thresh,
        #     clamp_min=-2.0, clamp_max=0.5,  # <-- key
        #     learn_logvar_after=1500,  # freeze σ for ~1.5k steps
        # )

        ##TODO: Check correctness for disc_factor ==0
        self.depth_loss = LPIPSWithDiscriminator(
            disc_start=10_000,              # ignored when disc_factor=0.0
            logvar_init=0.0,
            kl_weight=self.hparams.beta_kl_joint,  # will be overridden to 0.0 in joint branch
            pixelloss_weight=1.0,
            disc_num_layers=3,
            disc_in_channels=self.hparams.depth_in_channels,
            disc_factor=0.0,                # no GAN path
            disc_weight=1.0,
            perceptual_weight=1.0,
            use_actnorm=False,
            disc_conditional=False,
            disc_loss="hinge",
        )

        self.compiled = False

    # -------------------------
    # Setup / compile
    # -------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        if self.hparams.compile and stage in ["fit", "test"] and not self.compiled:
            self.event_ae = torch.compile(self.event_ae)
            self.depth_ae = torch.compile(self.depth_ae)
            self.compiled = True
            print(f"Compiled event & depth AutoencoderKLs with torch.compile for stage: {stage}")

    # -------------------------
    # Optimizer
    # -------------------------
    def configure_optimizers(self):
        params = []

        # encoders
        if not self.hparams.freeze_event_encoder:
            params += list(self.event_ae.encoder.parameters())
        if not self.hparams.freeze_depth_encoder:
            params += list(self.depth_ae.encoder.parameters())

        # decoders
        if not self.hparams.freeze_event_decoder:
            params += list(self.event_ae.decoder.parameters())
        if not self.hparams.freeze_depth_decoder:
            params += list(self.depth_ae.decoder.parameters())

        # quant/post-quant convs (both)
        params += list(self.event_ae.quant_conv.parameters())
        params += list(self.event_ae.post_quant_conv.parameters())
        params += list(self.depth_ae.quant_conv.parameters())
        params += list(self.depth_ae.post_quant_conv.parameters())

        # loss logvars
        params += [self.event_loss.logvar, self.depth_loss.logvar]

        optimizer = torch.optim.Adam(params, lr=self.hparams.lr)
        return {"optimizer": optimizer}

    def _stats(self, name: str, t: Optional[torch.Tensor]) -> str:
        if t is None:
            return f"{name}=None"
        return (f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, "
                f"min={t.min().item():.4g}, max={t.max().item():.4g}, "
                f"mean={t.mean().item():.4g}, std={t.std().item():.4g}")


    # Forward helpers (encode/decode/poe)
    @torch.no_grad()
    def _encode(self, x: torch.Tensor, which: str):
        if which == "event":
            return self.event_ae.encode(x)
        elif which == "depth":
            return self.depth_ae.encode(x)
        else:
            raise ValueError("which must be 'event' or 'depth'")

    def _decode(self, z: torch.Tensor, which: str):
        if which == "event":
            return self.event_ae.decode(z)
        elif which == "depth":
            return self.depth_ae.decode(z)
        else:
            raise ValueError("which must be 'event' or 'depth'")

    def _current_omega(self):
        # hyperparams: omega_start, omega_end=1.0, omega_anneal_steps
        s = getattr(self.hparams, "omega_anneal_steps", 5000)
        w0 = getattr(self.hparams, "omega_start", 0.5)
        w1 = getattr(self.hparams, "omega_end", 1.0)
        t = min(self.global_step, s) / max(1, s)
        return w0 + (w1 - w0) * t  # linear

    def _poe_from_present(self, posteriors: List[PoEPosterior], names: Optional[List[str]] = None) -> PoEPosterior:
        mus = [p.mean for p in posteriors]
        lvs = [p.logvar for p in posteriors]

        if self.hparams.use_prior_expert:
            mus.append(torch.zeros_like(mus[0]))
            lvs.append(torch.zeros_like(lvs[0]))

        mu, lv = poe_fuse(mus, lvs)
        return PoEPosterior(mu, lv)

    # Temporary override helper for depth KL per call
    def _depth_loss_call(self, *, inputs, reconstructions, posteriors, split: str, kl_weight: float,
                         optimizer_idx: int = 0, valid_mask: Optional[torch.Tensor] = None):
        # Temporarily override KL weight
        old_kw = float(self.depth_loss.kl_weight)
        self.depth_loss.kl_weight = kl_weight

        #TODO: Right now, it Clamp / freeze logvar to keep σ ∈ [exp(-2), exp(0)], whether clamping is needed?

        # with torch.no_grad():
        #     if self.global_step < 1500:  # freeze early
        #         self.depth_loss.logvar.fill_(0.0)
        #     else:
        #         self.depth_loss.logvar.clamp_(min=-2.0, max=0.0)

        try:
            # Apply masking if valid_mask is provided
            if valid_mask is not None:
                # Ensure mask has same spatial dimensions as inputs
                if valid_mask.ndim == 4 and inputs.ndim == 4:
                    # Expand mask to match channel dimension if needed
                    if valid_mask.size(1) != inputs.size(1):
                        valid_mask = valid_mask.expand_as(inputs)
                    
                    # Apply mask directly: only compute loss on valid pixels
                    masked_inputs = inputs * valid_mask.float()
                    masked_reconstructions = reconstructions * valid_mask.float()
                    
                    loss, log = self.depth_loss(
                        inputs=masked_inputs,
                        reconstructions=masked_reconstructions,
                        posteriors=posteriors,
                        optimizer_idx=optimizer_idx,  # generator branch
                        global_step=self.global_step,
                        last_layer=None,
                        cond=None,
                        split=split,
                        weights=None,
                    )
                else:
                    # Fallback: compute loss normally if mask dimensions don't match
                    loss, log = self.depth_loss(
                        inputs=inputs,
                        reconstructions=reconstructions,
                        posteriors=posteriors,
                        optimizer_idx=optimizer_idx,  # generator branch
                        global_step=self.global_step,
                        last_layer=None,
                        cond=None,
                        split=split,
                        weights=None,
                    )
            else:
                # No mask provided: compute loss normally
                loss, log = self.depth_loss(
                    inputs=inputs,
                    reconstructions=reconstructions,
                    posteriors=posteriors,
                    optimizer_idx=optimizer_idx,  # generator branch
                    global_step=self.global_step,
                    last_layer=None,
                    cond=None,
                    split=split,
                    weights=None,
                )
        finally:
            self.depth_loss.kl_weight = old_kw
        return loss, log


    def _get_event_and_depth(self, batch: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch = self._fill_missing_keys(batch)
        x_event, x_depth_in= self._generate_condition(batch)

        # Resize tensors and corresponding valid_mask if needed for VAE compatibility
        # KL-f4 VAE has downsampling factor of 4, so inputs need to be divisible by 4
        if self.hparams.allow_resize:
            if x_event is not None:
                x_event, batch = self._resize_tensor_and_mask(x_event, batch, "event")
            if x_depth_in is not None:
                x_depth_in, batch = self._resize_tensor_and_mask(x_depth_in, batch, "depth")

        #TODO: check whehter this is correct way of replicating channels
        if (
                x_depth_in is not None
                and self.hparams.depth_encoder_expects_rgb
                and x_depth_in.ndim == 4
                and x_depth_in.size(1) == 1
        ):
            x_depth_in = x_depth_in.repeat_interleave(3, dim=1)  # [B,1,H,W] -> [B,3,H,W]

        return x_event, x_depth_in

    def _resize_tensor_and_mask(self, tensor: torch.Tensor, batch: Any, modality: str) -> Tuple[torch.Tensor, Any]:
        """
        Resize tensor to be compatible with VAE (divisible by downsampling factor)
        and also resize corresponding valid_mask to maintain consistency.
        
        KL-f4 VAE has downsampling factor of 4, so we need dimensions divisible by 4.
        """
        from src.utils.event.event_data_utils import resize_to_multiple_of_16
        import torch.nn.functional as F
        
        # For KL-f4 VAE, we need divisible by 4, not 16
        # But let's keep the existing function for now and just ensure mask consistency
        original_shape = tensor.shape[-2:]
        resized_tensor = resize_to_multiple_of_16(tensor)  # This ensures divisible by 16 (overkill but safe)
        new_shape = resized_tensor.shape[-2:]
        
        # If tensor was resized, also resize the corresponding valid_mask
        if original_shape != new_shape and "valid_mask_raw" in batch:
            valid_mask = batch["valid_mask_raw"]
            if valid_mask is not None and valid_mask.shape[-2:] == original_shape:
                # Resize mask using nearest interpolation to preserve binary nature
                resized_mask = F.interpolate(
                    valid_mask.float(),
                    size=new_shape,
                    mode='nearest'
                ).bool()
                batch["valid_mask_raw"] = resized_mask
        
        return resized_tensor, batch

    def _log_selected(self, prefix: str, log_dict: Dict[str, torch.Tensor], keep: set):
        """Log only selected leaf keys from a '{split}/key' dict."""
        for k, v in log_dict.items():
            leaf = k.split("/")[-1]  # e.g., "total_loss"
            if leaf in keep:
                self.log(f"{prefix}/{leaf}", v, on_step=True, on_epoch=False)



    def training_step(self, batch: Any, batch_idx: int):
        x_event, x_depth = self._get_event_and_depth(batch)

        present = []
        post_list = []
        if x_event is not None:
            pe = self._encode(x_event, "event")
            if getattr(self.hparams, "omega_event", 1.0) != 1.0:
                delta = -torch.log(torch.tensor(self.hparams.omega_event, device=pe.logvar.device))
                pe = PoEPosterior(pe.mean, pe.logvar + delta)  # precision *= omega_event
            post_list.append(pe)
            present.append("event")
        if x_depth is not None:
            pd = self._encode(x_depth, "depth")
            if getattr(self.hparams, "omega_depth", 1.0) != 1.0:
                delta = -torch.log(torch.tensor(self.hparams.omega_depth, device=pd.logvar.device))
                pd = PoEPosterior(pd.mean, pd.logvar + delta)  # precision *= omega_depth
            post_list.append(pd)
            # post_list.append(PoEPosterior(pd.mean, pd.logvar))
            present.append("depth")

        assert len(present) >= 1, "At least one modality must be present in the batch."

        logs: Dict[str, torch.Tensor] = {}
        total = 0.0

        # joint term: recon (no KL inside) + ONE explicit KL(joint)
        joint_post = self._poe_from_present(post_list)
        z_joint = joint_post.sample()

        if x_event is not None:
            recon_e = self._decode(z_joint, "event")
            recon_e_loss, log_e = self.event_loss(
                inputs=x_event, reconstructions=recon_e, posteriors=joint_post,
                kl_weight=0.0, split="train" # no KL inside
            )

            #TODO: For debugging purposes
            B, C, H, W = x_event.shape
            pix = (B * C * H * W)
            if "train/event_nll_per_pixel" not in logs:
                nll_sum = log_e.get("train/nll_loss", log_e.get("nll_loss", None))
                if nll_sum is not None:
                    self.log("train/event/nll_per_pixel", nll_sum / max(1, pix), on_step=True, on_epoch=False)

            # # channel-wise active L1 to protect bin separation
            # active = (x_event.abs() > self.hparams.event_thresh).float()
            # act_cnt = active.sum()
            # l1_map = (recon_e - x_event).abs()
            # l1_active = (l1_map * active).sum() / act_cnt.clamp_min(1)
            # self.log("train/event/l1_active_mean", l1_active, on_step=True, on_epoch=False)


            total += self.hparams.lambda_joint_event * recon_e_loss
            self._log_selected("train/event", log_e, keep={"total_loss", "nll_loss"})

        if x_depth is not None:
            recon_d = self._decode(z_joint, "depth")
            # Get valid mask from batch
            valid_mask = batch.get("valid_mask_raw", None)
            recon_d_loss, log_d = self._depth_loss_call(
                inputs=x_depth, reconstructions=recon_d, posteriors=joint_post,
                split="train", kl_weight=0.0, optimizer_idx=0,  # no KL inside
                valid_mask=valid_mask
            )
            # total += recon_d_loss
            if x_depth is not None:
                total += self.hparams.lambda_joint_depth * recon_d_loss
            self._log_selected("train/depth", log_d, keep={"total_loss", "nll_loss"})

        # ONE joint KL
        kl_joint = joint_post.kl().sum(dim=list(range(1, joint_post.kl().ndim))).mean()
        total += self.hparams.beta_kl_joint * kl_joint
        self.log("train/kl_joint", kl_joint, on_step=True, on_epoch=False)

        # Unimodal auxiliary terms (MVAE-style)
        if self.hparams.include_unimodal_terms:
            # Event-only
            if x_event is not None:
                # pick the event posterior object from post_list/present
                p_event = None
                for p, name in zip(post_list, present):
                    if name == "event": p_event = p; break
                z_e = p_event.sample() if p_event is not None else z_joint
                recon_e_uni = self._decode(z_e, "event")
                loss_e_uni, log_eu = self.event_loss(
                    inputs=x_event, reconstructions=recon_e_uni, posteriors=(p_event or joint_post),
                    kl_weight=self.hparams.beta_kl_uni, split="train"
                )
                total += self.hparams.lambda_uni_event * loss_e_uni
                # self._log_selected("train/depth", log_d, keep={"total_loss", "rec_loss", "nll_loss", "sigma"})

            # Depth-only
            if x_depth is not None:
                p_depth = None
                for p, name in zip(post_list, present):
                    if name == "depth": p_depth = p; break
                z_d = p_depth.sample() if p_depth is not None else z_joint
                recon_d_uni = self._decode(z_d, "depth")
                loss_d_uni, log_du = self._depth_loss_call(
                    inputs=x_depth, reconstructions=recon_d_uni, posteriors=(p_depth or joint_post),
                    split="train", kl_weight=self.hparams.beta_kl_uni, optimizer_idx=0,
                    valid_mask=valid_mask
                )
                # total += loss_d_uni
                if x_depth is not None:
                    total += self.hparams.lambda_uni_depth * loss_d_uni
                # for k, v in log_du.items(): logs[f"train/depth_uni_{k.split('/')[-1]}"] = v

        # ##TODO: add Wassertain distance here to resolve distribution differences
        # mu_e, lv_e = pe.mean, pe.logvar
        # mu_d, lv_d = pd.mean, pd.logvar
        #
        # # Cheap & stable:
        # loss_w2 = w2_diag_gaussians(mu_e, lv_e, mu_d, lv_d)
        # lambda_w2 = getattr(self.hparams, "lambda_w2", 0.1)
        # total = total + lambda_w2 * loss_w2

        return total

    # -------------------------
    # Validation
    # -------------------------
    def validation_step(self, batch: Any, batch_idx: int):
        x_event, x_depth = self._get_event_and_depth(batch)
        # print("[VAL PREENC]", self._stats("x_event", x_event), self._stats("x_depth", x_depth))  # in validation_step

        present = []
        post_list = []
        if x_event is not None:
            pe = self._encode(x_event, "event"); present.append("event")
            if getattr(self.hparams, "omega_event", 1.0) != 1.0:
                delta = -torch.log(torch.tensor(self.hparams.omega_event, device=pe.logvar.device))
                pe = PoEPosterior(pe.mean, pe.logvar + delta)  # precision *= omega_event
            post_list.append(pe)
        if x_depth is not None:
            pd = self._encode(x_depth, "depth")
            if getattr(self.hparams, "omega_depth", 1.0) != 1.0:
                delta = -torch.log(torch.tensor(self.hparams.omega_depth, device=pd.logvar.device))
                pd = PoEPosterior(pd.mean, pd.logvar + delta)  # precision *= omega_depth
            post_list.append(pd)
            # post_list.append(PoEPosterior(pd.mean, pd.logvar))
            present.append("depth")

        joint_post = self._poe_from_present(post_list)
        z_joint = joint_post.sample()

        val_total = 0.0

        # Joint recon (no KL inside) + one KL
        if x_event is not None:
            recon_e = self._decode(z_joint, "event")
            recon_e_loss, log_e = self.event_loss(
                inputs=x_event, reconstructions=recon_e, posteriors=joint_post,
                kl_weight=0.0, split="val"
            )
            val_total += recon_e_loss
            for k, v in log_e.items(): self.log(f"val/event_{k.split('/')[-1]}", v, on_step=False, on_epoch=True, sync_dist=True)

            with torch.no_grad():
                mse = F.mse_loss(recon_e, x_event).item()
                mae = F.l1_loss(recon_e, x_event).item()
                dr = (x_event.max() - x_event.min())
                ## TODO: Be careful with dr
                psnr = (10 * torch.log10(dr** 2 / (mse + 1e-12))).item()
            self.log("val/event_mse", mse, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val/event_mae", mae, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val/event_psnr", psnr, on_step=False, on_epoch=True)

        if x_depth is not None:
            recon_d = self._decode(z_joint, "depth")
            # Get valid mask from batch
            valid_mask = batch.get("valid_mask_raw", None)
            recon_d_loss, log_d = self._depth_loss_call(
                inputs=x_depth, reconstructions=recon_d, posteriors=joint_post,
                split="val", kl_weight=0.0, optimizer_idx=0,
                valid_mask=valid_mask
            )
            # val_total += recon_d_loss
            if x_depth is not None:
                val_total += self.hparams.lambda_joint_depth * recon_d_loss

            for k, v in log_d.items(): self.log(f"val/depth_{k.split('/')[-1]}", v, on_step=False, on_epoch=True, sync_dist=True)

            with torch.no_grad():
                # Apply mask to metrics calculation too
                if valid_mask is not None:
                    # Expand mask to match depth channels if needed
                    if valid_mask.size(1) != x_depth.size(1):
                        valid_mask_exp = valid_mask.expand_as(x_depth)
                    else:
                        valid_mask_exp = valid_mask
                    
                    # Only compute metrics on valid pixels
                    valid_pixels = valid_mask_exp.bool()
                    if valid_pixels.any():
                        mse = F.mse_loss(recon_d[valid_pixels], x_depth[valid_pixels]).item()
                        mae = F.l1_loss(recon_d[valid_pixels], x_depth[valid_pixels]).item()
                        dr = (x_depth[valid_pixels].max() - x_depth[valid_pixels].min()).clamp_min(1e-12)
                        psnr = (10 * torch.log10(dr ** 2 / (mse + 1e-12))).item()
                    else:
                        mse = mae = psnr = 0.0
                else:
                    # Fallback: compute on all pixels
                    mse = F.mse_loss(recon_d, x_depth).item()
                    mae = F.l1_loss(recon_d, x_depth).item()
                    dr = (x_depth.max() - x_depth.min()).clamp_min(1e-12)
                    psnr = (10 * torch.log10(dr ** 2 / (mse + 1e-12))).item()
            self.log("val/depth_mse", mse, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val/depth_mae", mae, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val/depth_psnr", psnr, on_step=False, on_epoch=True, sync_dist=True)


        # ONE joint KL on validation
        kl_joint = joint_post.kl().sum(dim=list(range(1, joint_post.kl().ndim))).mean()
        self.log("val/kl_joint", kl_joint, on_step=False, on_epoch=True, sync_dist=True)
        val_total += self.hparams.beta_kl_joint * kl_joint

        self.log("val/total_loss", val_total, on_step=False, on_epoch=True, sync_dist=True)
        # >>> return modality tensors for PoELogger
        ret = {"val_loss": val_total}
        if x_event is not None:
            ret["event_input"] = x_event
            if 'recon_e' in locals():  # present if we decoded it
                ret["recon_event"] = recon_e
        if x_depth is not None:
            ret["depth_input"] = x_depth
            if 'recon_d' in locals():
                ret["recon_depth"] = recon_d
        
        # Pass valid_mask to logger for proper visualization masking
        if 'valid_mask' in locals() and valid_mask is not None:
            ret["valid_mask"] = valid_mask

##TODO: just trying to check the visualization result for e2d direct generation
        if pe is not None:
            z_e = PoEPosterior(pe.mean, pe.logvar)
            z_e = z_e.sample()
            ret["event2depth"] = self._decode(z_e, "depth")

        ret["posterior_event"] = pe
        ret["posterior_depth"] = pd
        return ret


    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        # 1) inputs
        x_event, x_depth = self._get_event_and_depth(batch)
        # print("[TEST PREENC]", self._stats("x_event", x_event), self._stats("x_depth", x_depth))
        
        # Get valid mask from batch for visualization
        valid_mask = batch.get("valid_mask_raw", None)

        outs: Dict[str, torch.Tensor] = {}
        if x_event is not None:
            outs["event_input"] = x_event
            pe = self._encode(x_event, "event")
            outs["prior_event"] = pe# q_e(z|e)
        else:
            pe = None

        if x_depth is not None:
            outs["depth_input"] = x_depth
            pd = self._encode(x_depth, "depth")
            outs["prior_depth"] = pd# q_d(z|d)
        else:
            pd = None

        # PoE posterior, include omega weights if set
        post_list = []
        if pe is not None:
            if getattr(self.hparams, "omega_event", 1.0) != 1.0:
                delta = -torch.log(torch.tensor(self.hparams.omega_event, device=pe.logvar.device))
                pe = PoEPosterior(pe.mean, pe.logvar + delta)  # precision *= omega_event
            post_list.append(pe)
        #directly generate event2depth to avoid depth leakage
        eventjoint_post = self._poe_from_present(post_list)
        z_eventjoint = eventjoint_post.sample()
        outs["event2depth"] = self._decode(z_eventjoint, "depth")
        
        if pd is not None:
            if getattr(self.hparams, "omega_depth", 1.0) != 1.0:
                delta = -torch.log(torch.tensor(self.hparams.omega_depth, device=pd.logvar.device))
                pd = PoEPosterior(pd.mean, pd.logvar + delta)  # precision *= omega_depth
            post_list.append(pd)

        # choose z construction exactly like validation; keep a flag to flip to mean if desired
        use_mean_for_recon = getattr(self.hparams, "use_mean_for_recon", False)

        if len(post_list) > 0:
            joint_post = self._poe_from_present(post_list)
            z_joint = joint_post.sample()

            # joint reconstructions
            if x_event is not None:
                recon_e = self._decode(z_joint, "event")
                outs["recon_event"] = recon_e

                try:
                    mse = F.mse_loss(recon_e, x_event).item()
                    mae = F.l1_loss(recon_e, x_event).item()
                    dr = (x_event.max() - x_event.min()).clamp_min(1e-12)
                    psnr = (10 * torch.log10(dr ** 2 / (mse + 1e-12))).item()
                    self.log("test/event_mse", mse, on_step=False, on_epoch=True, sync_dist=True)
                    self.log("test/event_mae", mae, on_step=False, on_epoch=True, sync_dist=True)
                    self.log("test/event_psnr", psnr, on_step=False, on_epoch=True, sync_dist=True)
                except Exception:
                    pass

            if x_depth is not None:
                recon_d = self._decode(z_joint, "depth")
                outs["recon_depth"] = recon_d

                try:
                    mse = F.mse_loss(recon_d, x_depth).item()
                    mae = F.l1_loss(recon_d, x_depth).item()
                    dr = (x_depth.max() - x_depth.min()).clamp_min(1e-12)
                    psnr = (10 * torch.log10(dr ** 2 / (mse + 1e-12))).item()
                    self.log("test/depth_mse", mse, on_step=False, on_epoch=True, sync_dist=True)
                    self.log("test/depth_mae", mae, on_step=False, on_epoch=True, sync_dist=True)
                    self.log("test/depth_psnr", psnr, on_step=False, on_epoch=True, sync_dist=True)
                except Exception:
                    pass

            # prior samples already generated above using clean event space

        # event2depth already generated above to avoid depth leakage
        
        # Pass valid_mask to logger for proper visualization masking
        if valid_mask is not None:
            outs["valid_mask"] = valid_mask

        return outs


    @torch.no_grad()
    def sample(self, batch: Any) -> Dict[str, torch.Tensor]:
        x_event, x_depth = self._get_event_and_depth(batch)
        present, post_list = [], []
        if x_event is not None:
            pe = self._encode(x_event, "event"); present.append("event")
            if getattr(self.hparams, "omega_event", 1.0) != 1.0:
                delta = -torch.log(torch.tensor(self.hparams.omega_event, device=pe.logvar.device))
                pe = PoEPosterior(pe.mean, pe.logvar + delta)  # precision *= omega_event
            post_list.append(pe)
        if x_depth is not None:
            pd = self._encode(x_depth, "depth")
            if getattr(self.hparams, "omega_depth", 1.0) != 1.0:
                delta = -torch.log(torch.tensor(self.hparams.omega_depth, device=pd.logvar.device))
                pd = PoEPosterior(pd.mean, pd.logvar + delta)  # precision *= omega_depth
            post_list.append(pd)
            # post_list.append(PoEPosterior(pd.mean, pd.logvar))
            present.append("depth")

        joint_post = self._poe_from_present(post_list)
        z = joint_post.sample()

        outs = {}
        if x_event is not None:
            outs["event_recon"] = self._decode(z, "event")
        if x_depth is not None:
            outs["depth_recon"] = self._decode(z, "depth")
        return outs

    @staticmethod
    def _maybe_freeze(module: nn.Module, do_freeze: bool):
        if do_freeze:
            for p in module.parameters():
                p.requires_grad = False

    @staticmethod
    def _clean_state_dict_keys(sd: dict,
                               root_prefixes: Tuple[str, ...] = (
                                   "autoencoder.", "first_stage_model.", "model.", "module.", "vae.",
                               )) -> dict:
        """Strip common top-level prefixes so keys match AutoencoderKL's modules."""
        cleaned = {}
        for k, v in sd.items():
            new_k = k
            for rp in root_prefixes:
                if new_k.startswith(rp):
                    new_k = new_k[len(rp):]
            cleaned[new_k] = v
        return cleaned

    @staticmethod
    def _load_full_autoencoder(ae: AutoencoderKL, ckpt_path: str, strict: bool = False):
        """
        Load FULL weights (encoder/decoder/quant/post_quant) into an AutoencoderKL.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        state = PoEVAELitModule._clean_state_dict_keys(state)

        wanted_prefixes = ("encoder.", "decoder.", "quant_conv.", "post_quant_conv.")
        filt = {k: v for k, v in state.items() if k.startswith(wanted_prefixes)}

        load_res = ae.load_state_dict(filt, strict=strict)
        missing = getattr(load_res, "missing_keys", [])
        unexpected = getattr(load_res, "unexpected_keys", [])
        print(f">> Loaded FULL AutoencoderKL from {ckpt_path} (strict={strict}) | "
              f"kept={len(filt)} missing={len(missing)} unexpected={len(unexpected)}")
        # if missing:
        #     print("   missing:", missing[:8], "..." if len(missing) > 8 else "")
        if unexpected:
            print("   unexpected:", unexpected[:8], "..." if len(unexpected) > 8 else "")
