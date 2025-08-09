
from typing import List, Tuple
import torch
import torch.nn.functional as F

def scale_invariant_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Eq (3) in the paper: L_si = (1/n) sum_u r(u)^2 - (1/n^2)(sum_u r(u))^2
    Expect pred and target to be normalized log-depth in [0,1].
    pred, target: [..., 1, H, W]
    mask: boolean or float mask of valid gt (same spatial dims)
    """
    r = pred - target
    if mask is not None:
        r = r * mask
        n = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    else:
        n = torch.tensor(r.shape[-2]*r.shape[-1], device=r.device, dtype=r.dtype)
        n = n.view([1]*(r.ndim-2) + [1,1])
    term1 = (r ** 2).sum(dim=(-2, -1), keepdim=True) / n
    term2 = (r.sum(dim=(-2, -1), keepdim=True) / n) ** 2
    return (term1 - term2).mean()

def gradient_x(img: torch.Tensor) -> torch.Tensor:
    return img[..., :, 1:] - img[..., :, :-1]

def gradient_y(img: torch.Tensor) -> torch.Tensor:
    return img[..., 1:, :] - img[..., :-1, :]

def multi_scale_grad_loss(pred: torch.Tensor, target: torch.Tensor, num_scales: int = 4, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Eq (4) in the paper (L1 on gradients of residuals) across scales.
    pred, target: [..., 1, H, W]
    Returns average over scales.
    """
    loss = 0.0
    p = pred
    t = target
    m = mask
    for _ in range(num_scales):
        r = p - t
        gx = torch.abs(gradient_x(r))
        gy = torch.abs(gradient_y(r))
        if m is not None:
            # shrink mask to match gradient shapes
            mx = m[..., :, 1:] * m[..., :, :-1]
            my = m[..., 1:, :] * m[..., :-1, :]
            gx = gx * mx
            gy = gy * my
            denom = (mx.sum(dim=(-2, -1)) + my.sum(dim=(-2, -1))).clamp_min(1.0)
            loss += (gx.sum(dim=(-2, -1)) + gy.sum(dim=(-2, -1))) / denom
        else:
            # mean over valid pixels
            n = (gx.numel() / gx.shape[0])  # per-batch element count
            loss += (gx.sum(dim=(-2, -1)) + gy.sum(dim=(-2, -1))) / n
        # downsample for next scale
        if p.shape[-1] > 1 and p.shape[-2] > 1:
            p = F.avg_pool2d(p, kernel_size=2, stride=2, ceil_mode=False)
            t = F.avg_pool2d(t, kernel_size=2, stride=2, ceil_mode=False)
            if m is not None:
                m = F.avg_pool2d(m, kernel_size=2, stride=2, ceil_mode=False)
                m = (m > 0.99).float()  # keep only fully valid pooled cells
    return (loss / num_scales).mean()

def si_plus_grad_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None, lambda_grad: float = 0.5, num_scales: int = 4) -> torch.Tensor:
    return scale_invariant_loss(pred, target, mask) + lambda_grad * multi_scale_grad_loss(pred, target, num_scales, mask)
