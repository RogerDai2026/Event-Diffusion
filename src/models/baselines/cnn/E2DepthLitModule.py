# file: src/models/baselines/e2d/e2d_lit.py

from typing import Any, Dict, Tuple, Optional
import torch
import numpy as np
import math
import torch.nn.functional as F
from src.models.event.generic_e2d_module import GenericE2DModule
from src.utils.event2depth_utils.model.metric import (
    abs_rel_diff as e2d_abs_rel,
    squ_rel_diff as e2d_sq_rel,
    rms_linear as e2d_rmse,
    scale_invariant_error as e2d_silog,
    mean_error as e2d_mean_err,
    median_error as e2d_median_err,
)

class E2DDepthLitModule(GenericE2DModule):
    """
    Lightning wrapper for E2Depth (E2VID/E2VIDRecurrent).
    Expects:
      - condition: [B, NBIN, H, W]
      - gt:        [B, 1,    H, W]  (normalized log-depth in [0,1])
    Pads H,W to multiples of 16, runs the net, then unpads.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Any,
        compile: bool = False,
        eval_min: float = 2.0,
        eval_max: float = 80.0,
        pad_multiple: int = 16,
        **kwargs: Any,
    ):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.pad_multiple = int(pad_multiple)
        self.eval_min = float(eval_min)
        self.eval_max = float(eval_max)
        self.alpha = float(-math.log(self.eval_min / self.eval_max + 1e-12))
        self.save_hyperparameters(logger=False, ignore=("net", "criterion", "optimizer"))
        self._optim_factory = optimizer  # optimizer is a partial/factory

        # accumulators
        self._reset_metric_sums()


    # ---------- E2D denormalization (no external deps) ----------
    @staticmethod
    def _clamp01(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, 1.0)

    def _denorm_log_e2d(self, hat: torch.Tensor) -> torch.Tensor:
        """hat∈[0,1] -> meters using E2D: D = D_max * exp(-alpha*(1-hat))."""
        hat = self._clamp01(hat)
        return self.eval_max * torch.exp(-self.alpha * (1.0 - hat))

    # ---------- metrics accumulation ---------
    def _reset_metric_sums(self):
        self._metric_sums = {k: 0.0 for k in
            ["abs_rel", "sq_rel", "rmse", "silog", "mean_err", "median_err"]}
        self._metric_count = 0

    def _to_eval_numpy(self, hat_norm: torch.Tensor) -> np.ndarray:
        """denorm -> meters, clamp to [d_min,d_max], return [B,H,W] numpy."""
        d = self._denorm_log_e2d(hat_norm)[:, 0]            # [B,H,W]
        d = torch.clamp(d, self.eval_min, self.eval_max)
        return d.detach().cpu().numpy()

    def _update_metrics_from_batch(self, pred_hat: torch.Tensor, gt_hat: torch.Tensor):
        p_np = self._to_eval_numpy(pred_hat)
        g_np = self._to_eval_numpy(gt_hat)
        B = p_np.shape[0]
        for i in range(B):
            p, g = p_np[i], g_np[i]
            self._metric_sums["abs_rel"]    += float(e2d_abs_rel(p, g))
            self._metric_sums["sq_rel"]     += float(e2d_sq_rel(p, g))
            self._metric_sums["rmse"]       += float(e2d_rmse(p, g))
            self._metric_sums["silog"]      += float(e2d_silog(p, g))
            self._metric_sums["mean_err"]   += float(e2d_mean_err(p, g))
            self._metric_sums["median_err"] += float(e2d_median_err(p, g))
            self._metric_count += 1

    def _finalize_and_log_metrics(self, prefix: str):
        if self._metric_count == 0:
            return
        avg = {f"{prefix}/{k}": self._metric_sums[k] / self._metric_count
               for k in self._metric_sums}
        for k, v in avg.items():
            self.log(k, v, prog_bar=False, sync_dist=True)  # DDP-safe mean
        self._reset_metric_sums()


    # ---------- utils ----------
    def _pad_to_multiple(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
        # x: [B,C,H,W]; returns padded tensor and (left,right,top,bottom) pad tuple
        _, _, H, W = x.shape
        m = self.pad_multiple
        pad_h = (m - H % m) % m
        pad_w = (m - W % m) % m
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0, 0, 0)
        # F.pad takes (left, right, top, bottom)
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        return x_pad, (0, pad_w, 0, pad_h)

    def _unpad(self, x: torch.Tensor, pad: Tuple[int,int,int,int]) -> torch.Tensor:
        if pad == (0,0,0,0):
            return x
        _, r, _, b = pad
        if r > 0:
            x = x[..., :, :-r]
        if b > 0:
            x = x[..., :-b, :]
        return x

    # ---------- PL hooks ----------
    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            print("[E2DDepthLit] Model compiled.")
        elif self.hparams.compile and stage == "test":
            print("[E2DDepthLit] Warning: torch.compile only runs during fit.")

    def configure_optimizers(self) -> Dict[str, Any]:
        optim = self._optim_factory(params=self.net.parameters())
        return {"optimizer": optim}

    # ---------- forward / sample ----------
    def _forward_padded(self, cond: torch.Tensor, prev_states: Optional[Any] = None) -> torch.Tensor:
        cond_pad, pad = self._pad_to_multiple(cond)
        # both E2VID and E2VIDRecurrent accept (x, prev_states)
        out = self.net(cond_pad, prev_states)
        if isinstance(out, tuple):
            out = out[0]  # (pred, states) -> pred
        out = self._unpad(out, pad)
        return out

    def forward(self, event: torch.Tensor, prev_states: Optional[Any] = None) -> torch.Tensor:
        return self._forward_padded(event, prev_states)

    def sample(self, condition: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._forward_padded(condition, None)

    # ---------- loss ----------
    def _compute_loss(self, pred: torch.Tensor, gt: torch.Tensor, cond: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        loss, _ = self.criterion(pred, gt, None, valid_mask)
        return loss
        # try:
        #     loss, _ = self.criterion(prediction=pred, target=gt, voxel=None, valid_mask=valid_mask)
        #     return loss
        # except TypeError:
        #     try:
        #         loss, _ = self.criterion(net=self.net, img_clean=gt, img_lr=cond)
        #         return loss
        #     except TypeError:
        #         return F.l1_loss(pred, gt)

    # ---------- steps ----------
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        batch = self._fill_missing_keys(batch)
        cond, gt = self._generate_condition(batch)
        mask = batch.get("valid_mask_raw", None)

        pred = self._forward_padded(cond, None)  # pad → net → unpad
        loss = self._compute_loss(pred, gt, cond, mask)

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False, batch_size=cond.shape[0])
        return {
            "loss": loss,
            "prediction": pred,
            "gt": gt,
            "condition": cond,
            "global_index": batch.get("global_index", None),
        }

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        batch = self._fill_missing_keys(batch)
        cond, gt = self._generate_condition(batch)
        mask = batch.get("valid_mask_raw", None)

        pred = self._forward_padded(cond, None)
        loss = self._compute_loss(pred, gt, cond, mask)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=cond.shape[0], sync_dist=True)
        self._update_metrics_from_batch(pred_hat=pred, gt_hat=gt)
        return {"batch_dict": batch, "condition": cond}

    def on_validation_epoch_end(self) -> None:
        self._finalize_and_log_metrics(prefix="val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        batch = self._fill_missing_keys(batch)
        cond, gt = self._generate_condition(batch)
        with torch.no_grad():
            pred = self.sample(cond)
        self._update_metrics_from_batch(pred_hat=pred, gt_hat=gt)
        return {"batch_dict": batch, "condition": cond}

    def on_test_epoch_end(self) -> None:
        self._finalize_and_log_metrics(prefix="test")