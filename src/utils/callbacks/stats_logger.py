# src/utils/callbacks/forward_stats_logger.py
import math
import torch
import torch.nn.functional as F
from lightning.pytorch import Callback
import wandb

def _exp_logger(trainer):
    return getattr(getattr(trainer, "logger", None), "experiment", None)


def _moments(x: torch.Tensor):
    x = x.float()
    mu = x.mean()
    var = x.var(unbiased=False)
    std = (var + 1e-8).sqrt()
    z = (x - mu) / (std + 1e-8)
    skew = (z**3).mean()
    kurt = (z**4).mean() - 3.0
    return dict(mu=mu, std=std, var=var, skew=skew, kurt=kurt)


def _as_sigma(logvar_or_std: torch.Tensor):
    # accepts scalar, [C], or full map; returns broadcastable [1,C,1,1] or [1,1,1,1]
    if logvar_or_std.ndim == 0:
        s = logvar_or_std
        if "log" in str(logvar_or_std):  # not reliable, but we’ll handle below anyway
            pass
    # assume it's logvar if author passes logvar; else it's std
    # we detect by checking if values are mostly small/negative
    v = logvar_or_std.detach()
    if v.ndim == 0:
        is_log = v.item() < 2.0  # heuristic; logvar usually ~ [-10, +3]
    else:
        is_log = v.float().median().item() < 2.0
    std = torch.exp(0.5 * logvar_or_std) if is_log else logvar_or_std

    if std.ndim == 0:
        std = std.view(1, 1, 1, 1)
    elif std.ndim == 1:
        std = std.view(1, -1, 1, 1)
    return std


class ForwardStatsLogger(Callback):
    """
    Reads distributions returned by the module on val/test and logs:
      - encoder posterior stats (mean_mu, mean_sigma) per modality
      - standardized residual stats (mean, var, kurt) per modality
      - suggested sigma temperature τ to make Var[r]≈1
    Expected keys in `outputs` from the module:
      - "event_input", "recon_event", "posterior_event"
      - "depth_input", "recon_depth", "posterior_depth"
    Each posterior should expose .mean and .logvar (or .std).
    """

    def __init__(self, log_every_n_batches: int = 1, phase: str = "val"):
        super().__init__()
        self.log_every_n_batches = log_every_n_batches
        self.phase = phase  # "val" or "test"

    # ---- utilities ---------------------------------------------------------

    @staticmethod
    def _latent_stats(posterior):
        if posterior is None:
            return None
        mu = getattr(posterior, "mean", None)
        if mu is None:
            return None

        if hasattr(posterior, "logvar") and posterior.logvar is not None:
            sigma = torch.exp(0.5 * posterior.logvar)  # <-- recompute from adjusted logvar
        else:
            return None

        # batch-average per-dim, then summarize
        mean_mu = mu.mean(dim=0)  # [Z]
        mean_sigma = sigma.mean(dim=0)  # [Z]
        return dict(
            mean_mu=mean_mu.mean().item(),
            std_mu=mean_mu.std().item(),
            mean_sigma=mean_sigma.mean().item(),
        )

    @staticmethod
    def _resid_stats(x, xhat, sigma_out: torch.Tensor | None):
        if x is None or xhat is None:
            return None
        if sigma_out is not None:
            r = (x - xhat) / (sigma_out + 1e-8)
        else:
            e = x - xhat
            est = e.flatten().float().std(unbiased=False).clamp_min(1e-8)
            r = e / est

        r = r.float()
        m = _moments(r)
        var = m["var"].item()
        tau = (1.0 / max(var, 1e-8)) ** 0.5  # σ' = τ·σ to make Var[r]≈1
        return dict(
            resid_mean=m["mu"].item(),
            resid_var=var,
            resid_kurt=m["kurt"].item(),
            tau_suggest=tau,
        )

    # ---- hooks -------------------------------------------------------------

    def _do_log(self, trainer, batch_idx):
        if self.log_every_n_batches <= 1:
            return True
        return (batch_idx % self.log_every_n_batches) == 0

    def _sigma_from_loss(self, pl_module):
        # Try to read the Gaussian σ used by your loss (scalar or [C])
        loss_obj = getattr(pl_module, "loss", None) or getattr(pl_module, "criterion", None)
        if loss_obj is not None and hasattr(loss_obj, "logvar"):
            return _as_sigma(loss_obj.logvar)
        if loss_obj is not None and hasattr(loss_obj, "sigma"):
            return _as_sigma(loss_obj.sigma)
        return None

    def _log_from_outputs(self, trainer, pl_module, outputs, batch_idx):
        if not isinstance(outputs, dict):
            return
        if not self._do_log(trainer, batch_idx):
            return
        exp = _exp_logger(trainer)
        if exp is None:
            return

        ev_in  = outputs.get("event_input")
        ev_rec = outputs.get("recon_event")
        dp_in  = outputs.get("depth_input")
        dp_rec = outputs.get("recon_depth")
        post_e = outputs.get("posterior_event")
        post_d = outputs.get("posterior_depth")

        # latent side
        for tag, post in (("event", post_e), ("depth", post_d)):
            s = self._latent_stats(post)
            if s is not None:
                exp.log({
                    f"{self.phase}/{tag}/latent/mean_mu": s["mean_mu"],
                    f"{self.phase}/{tag}/latent/mean_sigma": s["mean_sigma"],
                })

        # residual side
        sigma_out = self._sigma_from_loss(pl_module)  # broadcasted tensor or None

        def _broadcast(sig, like):
            if sig is None or like is None:
                return None
            if sig.size(1) == 1 or like.size(1) == 1:
                return sig
            # if σ is scalar or [1], ok; if [C] but C>like.C, truncate; if C<like.C, repeat
            Csig = sig.size(1)
            Clike = like.size(1)
            if Csig == Clike:
                return sig
            if Csig == 1:
                return sig
            if Csig > Clike:
                return sig[:, :Clike]
            return sig.repeat(1, Clike, 1, 1)

        sig_ev = _broadcast(sigma_out, ev_in) if sigma_out is not None else None
        sig_dp = _broadcast(sigma_out, dp_in) if sigma_out is not None else None

        for tag, x, xhat, sig in (("event", ev_in, ev_rec, sig_ev),
                                  ("depth", dp_in, dp_rec, sig_dp)):
            s = self._resid_stats(x, xhat, sig)
            if s is not None:
                exp.log({
                    f"{self.phase}/{tag}/resid/mean": s["resid_mean"],
                    f"{self.phase}/{tag}/resid/var":  s["resid_var"],
                    f"{self.phase}/{tag}/resid/kurt": s["resid_kurt"],
                    f"{self.phase}/{tag}/sigma_temperature_tau_suggest": s["tau_suggest"],
                })

    # Validation & Test hooks

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0):
        if self.phase != "val": return
        self._log_from_outputs(trainer, pl_module, outputs, batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0):
        if self.phase != "test": return
        self._log_from_outputs(trainer, pl_module, outputs, batch_idx)
