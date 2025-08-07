from typing import Any, Dict, Tuple
import torch
from hydra.utils import instantiate
from scipy.stats.tests.test_continuous_fit_censored import optimizer

from src.utils.event.event_data_utils import resize_to_multiple_of_16
from src.utils.helper import yprint
from src.models.event.generic_e2d_module import GenericE2DModule
from src.utils.corrdiff_utils.inference import regression_step_only, diffusion_step_batch
from physicsnemo.models.diffusion import UNet as PN_UNet
from hydra.utils import call


class CorrDiffEventLitModule(GenericE2DModule):
    """
    Adaptation of your CorrDiff downscaling module for event→depth.
    Behaves like the CNNLitModule but with your high-capacity UNet backbone.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Any,
        compile: bool,
        allow_resize: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.save_hyperparameters(
            logger=False,
            ignore=("net", "optimizer_config", "criterion"),
        )

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            print("Model compiled.")
        elif self.hparams.compile and stage == "test":
            print("Warning: torch.compile only runs during fit.")
        if self.hparams.allow_resize:
            yprint("Resizing data to multiples of 16 for UNet compatibility.")

    def configure_optimizers(self) -> Dict[str, Any]:
        optim = self.hparams.optimizer(params=self.net.parameters())
        return {"optimizer": optim}

    def forward(self, event: torch.Tensor, global_index: torch.Tensor = None) -> torch.Tensor:
        # event.shape == [B, C_event, H, W]
        B, C_event, H, W = event.shape
        C_out = self.net.img_out_channels  # should be 1 for depth
        # 1) create a zero "high-res" tensor
        hr = torch.zeros(B, C_out, H, W,
                         device=event.device,
                         dtype=event.dtype)
        # 2) call the PhysicsNemo UNet wrapper:
        #    forward(x=hr, img_lr=event, sigma=0.0)
        #    Only pass global_index if it's not None
        if global_index is not None:
            return self.net(hr, event, global_index=global_index, sigma=torch.tensor(0.0, device=event.device))
        else:
            return self.net(hr, event, sigma=torch.tensor(0.0, device=event.device))

    # , sigma = torch.tensor(0.0, device=event.device)

    def _maybe_resize(self, *tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if not self.hparams.allow_resize:
            return tensors
        return tuple(resize_to_multiple_of_16(t) for t in tensors)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        # batch comes from GenericE2DModule dataloader
        batch = self._fill_missing_keys(batch)
        cond, gt = self._generate_condition(batch)
        cond, gt = self._maybe_resize(cond, gt)
        global_index = batch.get("global_index", None)

        pred = self.forward(cond, global_index)
        loss, prediction = self.criterion(
            net=self.net,
            img_clean=gt,
            img_lr=cond,
        )
        loss = loss.mean()
        self.log(
            "train/loss", loss,
            on_step=True, on_epoch=False,
            prog_bar=False, batch_size=cond.shape[0]
        )
        return {
            "loss":       loss,
            "prediction": pred,   # shape (B, 1, H, W)
            "gt":         gt,           # shape (B, 1, H, W)
            "condition":  cond,         # shape (B, C_cond, H, W)
            "global_index": global_index,  # [B,2,H,W]
        }

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        batch = self._fill_missing_keys(batch)
        cond, gt = self._generate_condition(batch)
        cond, gt = self._maybe_resize(cond, gt)

        loss, prediction = self.criterion(
            net=self.net,
            img_clean=gt,
            img_lr=cond,
        )
        loss = loss.mean()
        self.log("val/loss", loss,
                 on_step=False, on_epoch=True,
                 prog_bar=False, batch_size=cond.shape[0],
                 sync_dist=True)
        step_output = {"batch_dict": batch, "condition": cond}
        return step_output

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        # 1) prepare inputs just like train/val
        batch = self._fill_missing_keys(batch)
        cond, gt = self._generate_condition(batch)
        cond, gt = self._maybe_resize(cond, gt)

        step_output = {"batch_dict": batch, "condition": cond}
        return step_output


    def sample(self, condition: torch.Tensor) -> torch.Tensor:
        # optionally resize here, if you want the same logic as train/val
        B, C, H, W = condition.shape
        latent_shape = (B, 1, H, W)
        # enforce the "regression" time-step t=1.0 and float64, as in the original
        return regression_step_only(
            net=self.net,
            img_lr=condition,
            latents_shape=latent_shape,
            lead_time_label=None
        )



class CorrDiffEventDiffusionModule(GenericE2DModule):
    """
    A LightningModule that
     1) loads a pretrained regression‐UNet,
     2) trains a residual‐diffusion UNet on top of its output,
     3) at test/sample time, sums mean + residual.

    Assumes:
    - self.net is the *residual* UNet (EDMPrecondSuperResolution or plain UNet wrapper).
    - regression_net_ckpt points to a .ckpt holding the regression‐UNet state_dict().
    - self.sampler is your diffusion sampler function.
    """

    def __init__(
            self,
            net: torch.nn.Module,
            criterion: Any,
            optimizer: Any,
            regression_net_ckpt: str,
            sampling,
            compile: bool = False,
            allow_resize: bool = True,
            **kwargs
    ):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.optimizer_cfg = optimizer
        self.regression_ckpt = regression_net_ckpt
        # self.sampling_cfg = sampling
        self.sampler = sampling
        self.compile = compile
        self.allow_resize = allow_resize

        # will be filled in setup()
        self.regression_net = None
        self.save_hyperparameters(logger=False,
                                  ignore=("net", "criterion", "optimizer_cfg", "sampling"))


    def setup(self, stage: str):
        if self.compile and stage == "fit":
            self.net = torch.compile(self.net)

        cfg = self.hparams.regression_model_cfg
        reg_net = PN_UNet(
            img_channels     = cfg["img_channels"],
            img_resolution   = cfg["img_resolution"],
            img_in_channels  = cfg["img_in_channels"],
            img_out_channels = cfg["img_out_channels"],
            use_fp16         = cfg.get("use_fp16", False),
            model_type       = cfg.get("model_type", "SongUNetPosEmbd"),
            **cfg.get("model_kwargs", {}),
        )

        ckpt = torch.load(self.regression_ckpt, map_location=self.device, weights_only=False)
        sd   = ckpt.get("state_dict", ckpt)
        stripped = {}
        for k, v in sd.items():
            if k.startswith("net.model."):
                stripped[k[len("net.model."):]] = v
            elif k.startswith("net."):
                stripped[k[len("net."):]] = v

        reg_net.model.load_state_dict(stripped, strict=False)

        reg_net.eval()
        for p in reg_net.parameters():
            p.requires_grad = False
        self.regression_net = reg_net.to(self.device)

        yprint(f"Loaded regression U-Net from {self.regression_ckpt}")

        self.criterion = self.criterion(self.regression_net)


    def configure_optimizers(self) -> Dict[str, Any]:
        return {"optimizer": self.optimizer_cfg(params=self.net.parameters())}

    def forward(self, event: torch.Tensor) -> torch.Tensor:
        # event: (B, C_event, H, W)
        B, _, H, W = event.shape
        # 1) zero‐initialize the "HR" channels
        hr = torch.zeros(B, self.net.img_out_channels, H, W,
                         device=event.device, dtype=event.dtype)
        # 2) feed through the diffusion UNet in regression‐only mode (σ=0)
        return self.net(hr, event, sigma=torch.tensor(0.0,
                                                      device=event.device,
                                                      dtype=torch.float32))

    def _maybe_resize(self, *tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if not self.allow_resize:
            return tensors
        return tuple(resize_to_multiple_of_16(t) for t in tensors)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch = self._fill_missing_keys(batch)
        event, gt = self._generate_condition(batch)  # event→depth
        event, gt = self._maybe_resize(event, gt)

        # criterion should return (loss, prediction)
        loss, _ = self.criterion(
            net=self.net,
            img_clean=gt,
            img_lr=event
        )
        loss = loss.mean()
        if torch.isnan(loss).any():
            # this will always show, even if you skip the batch
            n_nan = torch.isnan(loss).sum().item()
            n_inf = torch.isinf(loss).sum().item()
            print(f"[CRIT-DEBUG] Bad batch {batch_idx}: loss has {n_nan} NaNs and {n_inf} Infs → skipping")
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)

        self.log("train/loss", loss, on_step=True, on_epoch=False,
                 prog_bar=False, batch_size=event.shape[0])
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        batch = self._fill_missing_keys(batch)
        event, gt = self._generate_condition(batch)
        event, gt = self._maybe_resize(event, gt)

        loss, _ = self.criterion(
            net=self.net,
            img_clean=gt,
            img_lr=event
        )
        loss = loss.mean()
        self.log("val/loss", loss, on_step=False, on_epoch=True,
                 prog_bar=False, batch_size=event.shape[0], sync_dist=True)

        # pass these along for the EventLogger callback
        return {"batch_dict": batch, "condition": event}

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        # 1) prepare inputs just like train/val

        cond = batch['rgb_norm']
        # replace NaN, +inf, –inf with 0
        cond = torch.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0)
        batch['rgb_norm'] = cond

        batch = self._fill_missing_keys(batch)
        cond, gt = self._generate_condition(batch)
        cond, gt = self._maybe_resize(cond, gt)

        B, C, H, W = cond.shape
        latent_shape = (B, 1, H, W)

        # 2) run both stages under no_grad
        with torch.inference_mode():
            # a) regression mean
            mean_pred = regression_step_only(
                net=self.regression_net,
                img_lr=cond,
                latents_shape=latent_shape,
                lead_time_label=None
            )

            res_pred = diffusion_step_batch(
                net=self.net,
                sampler_fn=self.sampler,
                img_lr=cond,
                img_shape=(H, W),
                img_out_channels=1,
                device=self.device,
                hr_mean=None, #possibly mean_prediction
                lead_time_label=None,
            )
        final_pred = mean_pred + res_pred

        return {
            "condition":         cond.cpu(),
            "mean_prediction":   final_pred.cpu(),
            "ground_truth":      gt.cpu(),
            "batch_idx":         batch_idx,
        }

    def sample(self, event: torch.Tensor) -> torch.Tensor:
        # exactly the same two‐stage pipeline
        B, C, H, W = event.shape
        latent_shape = (B, 1, H, W)
            # a) regression mean
        mean_pred = regression_step_only(
            net=self.regression_net,
            img_lr=event,
            latents_shape=latent_shape,
            lead_time_label=None
            )
        res_pred = diffusion_step_batch(
                net=self.net,
                sampler_fn=self.sampler,
                img_lr=event,
                img_shape=(H, W),
                img_out_channels=1,
                device=self.device,
                hr_mean=mean_pred,
                lead_time_label=None
            )
        return mean_pred + res_pred

    #note: mgiht be wrong (noise should cover the mean_pred)
