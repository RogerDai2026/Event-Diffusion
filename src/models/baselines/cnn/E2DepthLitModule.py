
from typing import Any, Dict, Optional, Tuple
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from lightning.pytorch import LightningModule
except Exception:
    from pytorch_lightning import LightningModule

# Optional dependency — if present, we'll use your utils
from src.models.event.generic_e2d_module import GenericE2DModule


# --- Loss (paper-accurate) ---
def gradient_x(img: torch.Tensor) -> torch.Tensor:
    return img[..., :, 1:] - img[..., :, :-1]

def gradient_y(img: torch.Tensor) -> torch.Tensor:
    return img[..., 1:, :] - img[..., :-1, :]

def scale_invariant_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    r = pred - target
    if mask is not None:
        r = r * mask
        n = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    else:
        n = torch.tensor(r.shape[-2]*r.shape[-1], device=r.device, dtype=r.dtype).view([1]*(r.ndim-2)+[1,1])
    term1 = (r**2).sum(dim=(-2, -1), keepdim=True) / n
    term2 = (r.sum(dim=(-2, -1), keepdim=True) / n) ** 2
    return (term1 - term2).mean()

def multi_scale_grad_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor], num_scales: int) -> torch.Tensor:
    loss = 0.0
    p = pred
    t = target
    m = mask
    for _ in range(num_scales):
        r = p - t
        gx = torch.abs(gradient_x(r))
        gy = torch.abs(gradient_y(r))
        if m is not None:
            mx = m[..., :, 1:] * m[..., :, :-1]
            my = m[..., 1:, :] * m[..., :-1, :]
            gx = gx * mx
            gy = gy * my
            denom = (mx.sum(dim=(-2, -1)) + my.sum(dim=(-2, -1))).clamp_min(1.0)
            loss += (gx.sum(dim=(-2, -1)) + gy.sum(dim=(-2, -1))) / denom
        else:
            n = (gx.numel() / gx.shape[0])
            loss += (gx.sum(dim=(-2, -1)) + gy.sum(dim=(-2, -1))) / n
        # Downsample for next scale
        if p.shape[-2] > 1 and p.shape[-1] > 1:
            p = F.avg_pool2d(p, 2, 2)
            t = F.avg_pool2d(t, 2, 2)
            if m is not None:
                m = F.avg_pool2d(m, 2, 2)
                m = (m > 0.99).float()
    return (loss / num_scales).mean()

def si_plus_grad(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor], lambda_grad: float, num_scales: int) -> torch.Tensor:
    return scale_invariant_loss(pred, target, mask) + lambda_grad * multi_scale_grad_loss(pred, target, mask, num_scales)

def meters_to_log_normalized_depth(dm: torch.Tensor, Dmax: float, alpha: float) -> torch.Tensor:
    eps = 1e-6
    d = torch.clamp(dm, min=eps)
    out = 1.0 + (torch.log(d / Dmax) / alpha)
    return out.clamp(0.0, 1.0)

def log_normalized_to_meters(dhat: torch.Tensor, Dmax: float, alpha: float) -> torch.Tensor:
    return Dmax * torch.exp(-alpha * (1.0 - dhat))

class E2DepthImportLitModule(GenericE2DModule):
    """
    Zero-rewrite Lightning shim: imports the exact model class from the rpg_e2depth repo.

    Pass model_class_path like "e2depth.model.model.RecurrentUNet" and model_kwargs to match
    the official options. No reimplementation of the net is done here.
    """

    def __init__(
        self,
        model_class_path: str,
        model_kwargs: dict,
        optimizer_ctor,
        compile: bool = False,
        allow_resize: bool = True,
        lambda_grad: float = 0.5,
        num_scales: int = 4,
        Dmax: float = 80.0,
        alpha: float = 3.7,
        gt_is_metric: bool = True,
        clip_grad_norm: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__()
        # Dynamically import their model
        module_name, class_name = model_class_path.rsplit('.', 1)
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        self.net: nn.Module = cls(**(model_kwargs or {}))
        self.optimizer_ctor = optimizer_ctor
        self.save_hyperparameters(logger=False, ignore=("optimizer_ctor",))

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            try:
                self.net = torch.compile(self.net)
            except Exception as e:
                print(f"torch.compile failed: {e}")

    def configure_optimizers(self):
        return {"optimizer": self.optimizer_ctor(params=self.net.parameters())}

    # Optional resize hook
    def _maybe_resize(self, *tensors: torch.Tensor):
        if not self.hparams.allow_resize:
            return tensors
        try:
            from src.utils.event.event_data_utils import resize_to_multiple_of_16
            return tuple(resize_to_multiple_of_16(t) for t in tensors)
        except Exception:
            return tensors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Support recurrent nets with (x, state) -> (y, new_state) or feed-forward nets with x->y
        if hasattr(self.net, "forward"):
            try:
                # try recurrent signature
                if getattr(self, "_state", None) is None:
                    y, self._state = self.net(x, None)
                else:
                    y, self._state = self.net(x, self._state)
            except TypeError:
                # fall back to feed-forward
                y = self.net(x)
                self._state = None
        else:
            y = self.net(x)
            self._state = None
        return y

    def _prep_targets(self, gt: torch.Tensor) -> torch.Tensor:
        if self.hparams.gt_is_metric:
            return meters_to_log_normalized_depth(gt, self.hparams.Dmax, self.hparams.alpha)
        return gt.clamp(0.0, 1.0)

    def _step_common(self, batch: Dict[str, torch.Tensor]):
        batch = self._fill_missing_keys(batch)
        cond, gt = self._generate_condition(batch)
        cond, gt = self._maybe_resize(cond, gt)

        if cond.ndim == 5:
            # [B,L,C,H,W]
            B, L, C, H, W = cond.shape
            loss_acc = 0.0
            preds = []
            self._state = None
            for t in range(L):
                y = self.forward(cond[:, t])
                preds.append(y)
            pred = torch.stack(preds, dim=1)  # [B,L,1,H,W]
            gt_ln = self._prep_targets(gt)
            mask = torch.isfinite(gt_ln).float()
            for t in range(L):
                loss_acc = loss_acc + si_plus_grad(pred[:, t], gt_ln[:, t], mask[:, t], self.hparams.lambda_grad, self.hparams.num_scales)
            loss = loss_acc / L
            return loss, pred, gt_ln, cond

        elif cond.ndim == 4:
            # [B,C,H,W]
            self._state = None
            pred = self.forward(cond)
            gt_ln = self._prep_targets(gt)
            mask = torch.isfinite(gt_ln).float()
            loss = si_plus_grad(pred, gt_ln, mask, self.hparams.lambda_grad, self.hparams.num_scales)
            return loss, pred, gt_ln, cond
        else:
            raise ValueError(f"Unexpected cond shape: {cond.shape}")

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, _, _, cond = self._step_common(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=cond.shape[0])
        if self.hparams.clip_grad_norm and self.hparams.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.hparams.clip_grad_norm)
        return {"loss": loss}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, _, _, cond = self._step_common(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=cond.shape[0], sync_dist=True)
        return {"loss": loss}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        batch = self._fill_missing_keys(batch)
        cond, gt = self._generate_condition(batch)
        cond, gt = self._maybe_resize(cond, gt)
        return {"batch_dict": batch, "condition": cond}
