from typing import Any, Dict, Tuple
import torch
from hydra.utils import instantiate
from lightning.pytorch import LightningModule
from src.utils.event.event_data_utils import resize_to_multiple_of_16
from src.utils.helper import yprint
from src.models.event.generic_e2d_module import GenericE2DModule
from src.utils.corrdiff_utils.inference import regression_step_only, diffusion_step_batch
from physicsnemo.models.diffusion.unet import UNet as PN_UNet


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

    def forward(self, event: torch.Tensor) -> torch.Tensor:
        # event.shape == [B, C_event, H, W]
        B, C_event, H, W = event.shape
        C_out = self.net.img_out_channels  # should be 1 for depth
        # 1) create a zero “high-res” tensor
        hr = torch.zeros(B, C_out, H, W,
                         device=event.device,
                         dtype=event.dtype)
        # 2) call the PhysicsNemo UNet wrapper:
        #    forward(x=hr, img_lr=event, sigma=0.0)
        #    (or omit sigma if you’re using the regression-UNet forward which doesn’t take it)
        return self.net(hr, event, sigma=torch.tensor(0.0, device=event.device))

    def _maybe_resize(self, *tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if not self.hparams.allow_resize:
            return tensors
        return tuple(resize_to_multiple_of_16(t) for t in tensors)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # batch comes from GenericE2DModule dataloader
        batch = self._fill_missing_keys(batch)
        cond, gt = self._generate_condition(batch)
        cond, gt = self._maybe_resize(cond, gt)

        pred = self.forward(cond)
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
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        batch = self._fill_missing_keys(batch)
        cond, gt = self._generate_condition(batch)
        cond, gt = self._maybe_resize(cond, gt)

        pred = self.forward(cond)
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

        # 2) run regression-only inference
        B, C, H, W = cond.shape
        latent_shape = (B, 1, H, W)
        with torch.inference_mode():
            mean_pred = regression_step_only(
                net=self.net,
                img_lr=cond,
                latents_shape=latent_shape,
                lead_time_label=None
            )

        return {
            "condition": cond.cpu(),
            "mean_prediction": mean_pred.cpu(),
            "ground_truth": gt.cpu(),
            "batch_idx": batch_idx
        }

    def sample(self, condition: torch.Tensor) -> torch.Tensor:
        # optionally resize here, if you want the same logic as train/val
        B, C, H, W = condition.shape
        latent_shape = (B, 1, H, W)
        # enforce the “regression” time-step t=1.0 and float64, as in the original
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
        self.sampler = sampling
        self.compile = compile
        self.allow_resize = allow_resize

        # will be filled in setup()
        self.regression_net = None
        self.save_hyperparameters(logger=False,
                                  ignore=("net", "criterion", "optimizer_cfg", "sampler"))


    def setup(self, stage: str):
        # 1) Optionally compile your *residual* UNet
        if self.compile and stage == "fit":
            self.net = torch.compile(self.net)

        # 2) Reconstruct the *regression* U-Net
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

        # 3) Load & strip the Lightning checkpoint
        ckpt = torch.load(self.regression_ckpt, map_location="cpu")
        sd   = ckpt.get("state_dict", ckpt)
        stripped = {}
        for k, v in sd.items():
            if k.startswith("net.model."):
                stripped[k[len("net.model."):]] = v
            elif k.startswith("net."):
                stripped[k[len("net."):]] = v

        reg_net.model.load_state_dict(stripped, strict=False)

        # 4) Freeze & move to device
        reg_net.eval()
        for p in reg_net.parameters():
            p.requires_grad = False
        self.regression_net = reg_net.to(self.device)

        yprint(f"Loaded regression U-Net from {self.regression_ckpt}")

            # 5) finally bind it into your residual‐diffusion loss
            #    so that later your __call__ uses a real nn.Module, not a dict
        self.criterion = self.criterion(self.regression_net)

    def configure_optimizers(self) -> Dict[str, Any]:
        return {"optimizer": self.optimizer_cfg(params=self.net.parameters())}

    def forward(self, event: torch.Tensor) -> torch.Tensor:
        # event: (B, C_event, H, W)
        B, _, H, W = event.shape
        # 1) zero‐initialize the “HR” channels
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

    def test_step(self, batch: Tuple[Any], batch_idx: int) -> Dict[str, Any]:
        batch_dict, batch_coords, _, valid_mask = batch
        event, gt = self._generate_condition(batch_dict)

        # possibly ensemble
        if self.hparams.num_samples > 1 and event.shape[0] == 1:
            event = event.repeat(self.hparams.num_samples, 1, 1, 1)

        # 1) regression mean
        latent_shape = (event.shape[0], 1, event.shape[2], event.shape[3])
        mean_pred = regression_step_only(
            net=self.regression_net,
            img_lr=event,
            latents_shape=latent_shape,
            lead_time_label=None
        )

        # 2) residual via diffusion UNet
        res_pred = diffusion_step_batch(
            net=self.net,
            sampler_fn=self.sampler,
            img_lr=event,
            img_shape=(event.shape[2], event.shape[3]),
            img_out_channels=1,
            device=self.device,
            hr_mean=None,
            lead_time_label=None,
        )

        final = mean_pred + res_pred
        # stash into batch_dict for callbacks
        batch_dict["precip_output" if self.hparams.num_samples == 1 else
        [f"precip_output_{i}" for i in range(self.hparams.num_samples)]] = final

        return {
            "batch_dict": batch_dict,
            "batch_coords": batch_coords,
            "xr_low_res_batch": None,
            "valid_mask": valid_mask
        }

    def sample(self, event: torch.Tensor) -> torch.Tensor:
        # exactly the same two‐stage pipeline
        B, C, H, W = event.shape
        latent_shape = (B, 1, H, W)
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
            hr_mean=None,
            lead_time_label=None
        )
        return mean_pred + res_pred
