# Author: Bingxin Ke
# Last modified: 2024-04-18

import torch
import logging
import numpy as np


def get_depth_normalizer(cfg_normalizer):
    if cfg_normalizer is None:

        def identical(x):
            return x

        depth_transform = identical

    elif "scale_shift_depth" == cfg_normalizer.type:
        depth_transform = ScaleShiftDepthNormalizer(
            norm_min=cfg_normalizer.norm_min,
            norm_max=cfg_normalizer.norm_max,
            min_max_quantile=cfg_normalizer.min_max_quantile,
            clip=cfg_normalizer.clip,
        )
    elif "reversible_normalization" == cfg_normalizer.type:
        depth_transform = ScaleShiftDepthNormalizer(
            norm_min=cfg_normalizer.norm_min,
            norm_max=cfg_normalizer.norm_max,
            min_max_quantile=cfg_normalizer.min_max_quantile,
            clip=cfg_normalizer.clip,
            is_absolute=True,
            max_depth=cfg_normalizer.max_depth,
        )
    elif "log_inverse_normalization" == cfg_normalizer.type:
        depth_transform = ScaleShiftDepthNormalizer(
            norm_min=cfg_normalizer.norm_min,
            norm_max=cfg_normalizer.norm_max,
            min_max_quantile=cfg_normalizer.min_max_quantile,
            clip=cfg_normalizer.clip,
            is_absolute=True,
            inv_log=True,
            max_depth=cfg_normalizer.max_depth,
        )
    elif "log_depth" == cfg_normalizer.type:
        # Event2Depth-style log-depth target with explicit log-depth handling
        depth_transform = ScaleShiftDepthNormalizer(
            norm_min=cfg_normalizer.norm_min,
            norm_max=cfg_normalizer.norm_max,
            min_max_quantile=cfg_normalizer.min_max_quantile,
            clip=cfg_normalizer.clip,
            is_absolute=True,
            max_depth=cfg_normalizer.max_depth,
            log_depth=True,
        )
    elif "e2d_depth" == cfg_normalizer.type:  # E2Depth-style
        d_min = getattr(cfg_normalizer, "d_min", 2.0)
        d_max = getattr(cfg_normalizer, "max_depth", 80.0)
        return Event2DepthLogNormalizer(d_min=d_min, d_max=d_max)
    else:
        raise NotImplementedError
    return depth_transform

class DepthNormalizerBase:
    is_absolute = None
    far_plane_at_max = None

    def __init__(
        self,
        norm_min=-1.0,
        norm_max=1.0,
    ) -> None:
        self.norm_min = norm_min
        self.norm_max = norm_max
        raise NotImplementedError

    def __call__(self, depth, valid_mask=None, clip=None):
        raise NotImplementedError

    def denormalize(self, depth_norm, **kwargs):
        # For metric depth: convert prediction back to metric depth
        # For relative depth: convert prediction to [0, 1]
        raise NotImplementedError


class ScaleShiftDepthNormalizer(DepthNormalizerBase):
    """
    Use near and far plane to linearly normalize depth,
        i.e. d' = d * s + t,
        where near plane is mapped to `norm_min`, and far plane is mapped to `norm_max`
    Near and far planes are determined by taking quantile values.
    """


    def __init__(
        self, norm_min=-1.0, norm_max=1.0, min_max_quantile=0.02, 
        clip=True, is_absolute = False, far_plane_at_max = True,
        max_depth = 250.0,
        inv_log = False, log_depth = False,
    ) -> None:
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.norm_range = self.norm_max - self.norm_min
        self.min_quantile = min_max_quantile
        self.max_quantile = 1.0 - self.min_quantile
        self.clip = clip
        self.is_absolute = is_absolute
        self.max_depth = max_depth
        self.inv_log = inv_log
        self.use_log_depth = log_depth
        # self.far_plane_at_max = far_plane_at_max

    def __call__(self, depth_linear, valid_mask=None, clip=None):
        clip = clip if clip is not None else self.clip

        if valid_mask is None:
            valid_mask = torch.ones_like(depth_linear).bool()
        valid_mask = valid_mask & (depth_linear > 0)

        if self.use_log_depth:
            # If using log-depth target, delegate to log_normalization
            return self.log_normalization(depth_linear, valid_mask=valid_mask)

        if (True == self.inv_log):
            _min = np.log(1/(self.max_depth+1e-6))
            _max = np.log(1/(1e-6))
            _depth_linear = torch.log(1/(depth_linear+1e-6))
        elif (True == self.is_absolute):
            _min = 0
            _max = self.max_depth
            _depth_linear = depth_linear
        else:
            # Take quantiles as min and max
            _min, _max = torch.quantile(
                depth_linear[valid_mask],
                torch.tensor([self.min_quantile, self.max_quantile]),
            )
            _depth_linear = depth_linear

        # scale and shift
        depth_norm_linear = (_depth_linear - _min) / (
            _max - _min
        ) * self.norm_range + self.norm_min

        if clip:
            depth_norm_linear = torch.clip(
                depth_norm_linear, self.norm_min, self.norm_max
            )

        return depth_norm_linear

    def scale_back(self, depth_norm):
        # scale to [0, 1]
        depth_linear = (depth_norm - self.norm_min) / self.norm_range
        return depth_linear

    def denormalize(self, depth_norm, **kwargs):

        if self.use_log_depth:
            # log-depth inverse (Event2Depth) -> recover metric depth
            d_min = kwargs.get("d_min", None)
            d_max = kwargs.get("d_max", None)
            return self.log_denormalize(depth_norm, d_min=d_min, d_max=d_max)

        # reverse all the operations done to normalize (only or reversible normalization)
        if (not self.is_absolute):
            logging.warning(f"{self.__class__} only for reversible normalization")
            return self.scale_back(depth_norm=depth_norm)
        else:
            depth = (depth_norm - self.norm_min) / self.norm_range * (self.max_depth - 0) + 0
            return depth

    def log_normalization(self, depth_linear, d_min=None, d_max=None, valid_mask=None):
        """
        Event2Depth-style log-depth target:
            hat_D = 1 + (1/alpha) * ln(depth / D_max)
        where alpha = -ln(D_min / D_max).
        Returned value is clipped to [0,1] then linearly mapped to [norm_min, norm_max].
        """
        eps = 1e-6
        if valid_mask is None:
            valid_mask = torch.ones_like(depth_linear, dtype=torch.bool)
        valid_mask = valid_mask & (depth_linear > 0)

        # Determine D_min and D_max as tensors
        if self.is_absolute:
            D_max_val = d_max if d_max is not None else self.max_depth
            D_min_val = d_min if d_min is not None else 2.0
            D_max = torch.as_tensor(D_max_val, device=depth_linear.device, dtype=depth_linear.dtype)
            D_min = torch.as_tensor(D_min_val, device=depth_linear.device, dtype=depth_linear.dtype)
            # ensure numerically safe
            D_min = D_min.clamp_min(eps)
            D_max = D_max.clamp_min(D_min + eps)
        else:
            # infer from quantiles of this sample
            q = torch.tensor([self.min_quantile, self.max_quantile], device=depth_linear.device,
                             dtype=depth_linear.dtype)
            D_min, D_max = torch.quantile(
                depth_linear[valid_mask],
                q,
            )
            D_min = D_min.clamp_min(eps)
            D_max = D_max.clamp_min(D_min + eps)

        # compute alpha
        alpha = -torch.log(D_min / D_max + eps)

        # clamp depth to [D_min, D_max]
        depth_clamped = torch.clamp(depth_linear, min=D_min, max=D_max)

        # log-depth target
        hat_D = 1.0 + (1.0 / alpha) * torch.log(depth_clamped / D_max + eps)

        # clip to [0,1]
        hat_D = torch.clamp(hat_D, 0.0, 1.0)

        # map to [norm_min, norm_max]
        depth_norm = hat_D * self.norm_range + self.norm_min
        return depth_norm

    def log_denormalize(self, hat_D_norm, d_min=None, d_max=None):
        """
        Inverse of log_normalization: recover metric depth D^m from the log-depth target.
        hat_D_norm is in the normalized log-depth space (before mapping to [norm_min, norm_max]).
        """
        eps = 1e-6
        # Undo mapping to [0,1]:
        hat_D = (hat_D_norm - self.norm_min) / self.norm_range
        hat_D = torch.clamp(hat_D, 0.0, 1.0)

        # Determine D_min and D_max
        if self.is_absolute:
            D_max_val = d_max if d_max is not None else self.max_depth
            D_min_val = d_min if d_min is not None else 2.0
            # convert to tensors matching input
            D_max = torch.as_tensor(D_max_val, device=hat_D.device, dtype=hat_D.dtype)
            D_min = torch.as_tensor(D_min_val, device=hat_D.device, dtype=hat_D.dtype)
        else:
            raise ValueError("log_denormalize for relative mode requires explicit D_min/D_max or use absolute mode.")

        alpha = -torch.log(D_min / D_max + eps)

        # Recover metric depth
        Dm = D_max * torch.exp(-alpha * (1.0 - hat_D))
        return Dm

# Add this class next to your other normalizers
class Event2DepthLogNormalizer(DepthNormalizerBase):
    """
    E2Depth log-depth target (fixed constants, output in [0, 1]).
    Paper: D^m = D_max * exp(-alpha * (1 - D_hat)),
           with D_max = 80 m, alpha = 3.7 (i.e., D_min ≈ 2 m).
    """
    def __init__(self, d_min: float = 2.0, d_max: float = 80.0):
        self.norm_min = 0.0
        self.norm_max = 1.0
        self.norm_range = 1.0
        self.is_absolute = True
        self.d_min = float(d_min)
        self.d_max = float(d_max)
        # fixed alpha per paper (equivalently: -ln(d_min/d_max))
        self.alpha = float(-np.log(self.d_min / self.d_max + 1e-12))

    def __call__(self, depth, valid_mask=None, clip=True):
        eps = 1e-6
        if valid_mask is None:
            valid_mask = (depth > 0)
        # clamp to [d_min, d_max] as in the paper’s intent
        d = torch.clamp(depth, min=self.d_min, max=self.d_max)
        # hat_D in [0,1]
        hat = 1.0 + (1.0 / self.alpha) * torch.log(d / self.d_max + eps)
        hat = torch.clamp(hat, 0.0, 1.0)
        return hat  # already [0,1], no extra scaling

    def denormalize(self, hat_D_norm, **kwargs):
        # hat_D already in [0,1]
        hat = torch.clamp(hat_D_norm, 0.0, 1.0)
        return self.d_max * torch.exp(-self.alpha * (1.0 - hat))
