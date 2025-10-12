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
    delta_125  as delta_1,
    delta_125_2 as delta_2,
    delta_125_3 as delta_3,
    rmse_log as e2d_rmse_log,
    scale_invariant_log as e2d_si_log,
    abs_err_10m as abs_10,
    abs_err_20m as abs_20,
    abs_err_30m as abs_30,
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
            ["abs_rel", "sq_rel", "rmse", "rmse_log", "si_log", "delta_1", "delta_2", "delta_3", "abs_10", "abs_20","abs_30"]}
        # "mean_err", "median_err"
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
            self._metric_sums["rmse_log"]   += float(e2d_rmse_log(p, g))
            self._metric_sums["si_log"]     += float(e2d_si_log(p, g))
            self._metric_sums["delta_1"]    += float(delta_1(p, g))
            self._metric_sums["delta_2"]    += float(delta_2(p, g))
            self._metric_sums["delta_3"]    += float(delta_3(p, g))
            self._metric_sums["abs_10"]     += float(abs_10(p, g))
            self._metric_sums["abs_20"]     += float(abs_20(p, g))
            self._metric_sums["abs_30"]     += float(abs_30(p, g))
            # self._metric_sums["mean_err"]   += float(e2d_mean_err(p, g))
            # self._metric_sums["median_err"] += float(e2d_median_err(p, g))
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
    def _pad_to_multiple(self, x):
        B, C, H, W = x.shape
        m = self.pad_multiple
        pad_h = (m - H % m) % m
        pad_w = (m - W % m) % m
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0, 0, 0)
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
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
    def _forward_padded(self, cond, prev_states=None):
        cond_pad, pad = self._pad_to_multiple(cond)
        out = self.net(cond_pad, prev_states)
        if isinstance(out, tuple):
            out = out[0]
        out = self._unpad(out, pad)
        return out

    def forward(self, event: torch.Tensor, prev_states: Optional[Any] = None) -> torch.Tensor:
        return self._forward_padded(event, prev_states)

    def sample(self, condition: torch.Tensor) -> torch.Tensor:
        return self._forward_padded(condition, None)

    # ---------- loss ----------
    def _compute_loss(self, pred: torch.Tensor, gt: torch.Tensor, cond: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        loss, _ = self.criterion(pred, gt, None, valid_mask)
        return loss

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
        pred = self.sample(cond)
        self._update_metrics_from_batch(pred_hat=pred, gt_hat=gt)
        return {"batch_dict": batch, "condition": cond}

    def on_test_epoch_end(self) -> None:
        self._finalize_and_log_metrics(prefix="test")