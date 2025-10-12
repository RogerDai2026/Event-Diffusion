# alignment_losses.py
from __future__ import annotations
from typing import Iterable, Optional, Tuple, Sequence
import torch
import torch.nn.functional as F

def _as_tuple(x: Optional[Sequence[int]]) -> Tuple[int, ...]:
    return tuple() if x is None else tuple(x)

def _std_from_logvar(logvar: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Stable std from log-variance."""
    return (0.5 * logvar).exp().clamp_min(eps)

@torch.no_grad()
def _rand_unit_vectors(d: int, R: int, device=None, dtype=None, gaussian: bool = True) -> torch.Tensor:
    """Draw R random unit directions in R^d and L2-normalize them."""
    if gaussian:
        u = torch.randn(R, d, device=device, dtype=dtype)
    else:
        u = torch.empty(R, d, device=device, dtype=dtype).uniform_(-1.0, 1.0)
    u = F.normalize(u, dim=-1, eps=1e-12)
    return u

def w2_diag_gaussians(
    mu_e: torch.Tensor,
    logvar_e: torch.Tensor,
    mu_d: torch.Tensor,
    logvar_d: torch.Tensor,
    dims: Optional[Iterable[int]] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Closed-form squared W2 distance between diagonal Gaussians:
        W2^2 = ||mu_e - mu_d||^2 + ||sigma_e - sigma_d||^2,
    where sigma = exp(0.5 * logvar).

    Shapes:
        mu_*, logvar_*: [B, ...] (e.g., [B,Z] or [B,C,H,W]); shapes must match.

    Reduction:
        Averages over all non-batch dims (if any), then over batch → scalar tensor.
    """
    assert mu_e.shape == mu_d.shape == logvar_e.shape == logvar_d.shape, "shape mismatch"
    sigma_e = _std_from_logvar(logvar_e, eps)
    sigma_d = _std_from_logvar(logvar_d, eps)

    diff_mu2 = (mu_e - mu_d).pow(2)
    diff_sig2 = (sigma_e - sigma_d).pow(2)

    if dims is None:
        dims = tuple(range(1, mu_e.ndim)) if mu_e.ndim >= 2 else tuple(range(mu_e.ndim))
    else:
        dims = tuple(dims)

    w2 = (diff_mu2 + diff_sig2)
    if len(dims) > 0:
        w2 = w2.mean(dim=dims)
    return w2.mean()

def swd_from_params(
    mu_e: torch.Tensor,
    logvar_e: torch.Tensor,
    mu_d: torch.Tensor,
    logvar_d: torch.Tensor,
    K: int = 64,
    R: int = 128,
    dims: Optional[Iterable[int]] = None,
    eps: float = 1e-8,
    detach_dirs: bool = True,
    gaussian_dirs: bool = True,
) -> torch.Tensor:
    """
    Sliced-Wasserstein distance between two diagonal-Gaussian posteriors via sampling.

    Steps:
      1) Sample K latent codes from each posterior (reparameterization).
      2) Flatten latent dims to D.
      3) Draw R random unit directions in R^D.
      4) Project, sort along K, compute mean squared gap of order stats.
      5) Average over directions and batch → scalar tensor.

    Args:
        K: samples per posterior per batch element.
        R: number of random directions.
        detach_dirs: if True, do not backprop through random directions.
        gaussian_dirs: sample directions from N(0,I) then normalize (else uniform box).

    Shapes:
        mu_*, logvar_*: [B, ...] (e.g., [B,Z] or [B,C,H,W]); shapes must match.
    """
    assert mu_e.shape == mu_d.shape == logvar_e.shape == logvar_d.shape, "shape mismatch"

    if dims is None:
        dims = tuple(range(1, mu_e.ndim)) if mu_e.ndim >= 2 else tuple(range(mu_e.ndim))
    else:
        dims = tuple(dims)

    def _sample(mu, logvar):
        sigma = _std_from_logvar(logvar, eps)
        mu_s = mu.unsqueeze(1)          # [B,1,...]
        sigma_s = sigma.unsqueeze(1)     # [B,1,...]
        eps_s = torch.randn((mu_s.shape[0], K) + mu_s.shape[2:], device=mu.device, dtype=mu.dtype)
        z = mu_s + sigma_s * eps_s       # [B,K,...]
        z = z.flatten(start_dim=2)       # [B,K,D]
        return z

    z_e = _sample(mu_e, logvar_e)       # [B,K,D]
    z_d = _sample(mu_d, logvar_d)       # [B,K,D]
    D = z_e.shape[-1]

    dirs = _rand_unit_vectors(D, R, device=z_e.device, dtype=z_e.dtype, gaussian=gaussian_dirs)
    if not detach_dirs:
        dirs = dirs.requires_grad_(True)

    # Projections: [B,K,R]
    proj_e = torch.einsum("bkd,rd->bkr", z_e, dirs)
    proj_d = torch.einsum("bkd,rd->bkr", z_d, dirs)

    # Sort along samples K and compute mean squared gap of order stats
    proj_e_sorted, _ = torch.sort(proj_e, dim=1)
    proj_d_sorted, _ = torch.sort(proj_d, dim=1)
    sw2 = (proj_e_sorted - proj_d_sorted).pow(2).mean(dim=1)  # [B,R]

    return sw2.mean()
