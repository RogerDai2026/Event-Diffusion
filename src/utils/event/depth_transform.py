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
        inv_log = False
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
        # self.far_plane_at_max = far_plane_at_max

    def __call__(self, depth_linear, valid_mask=None, clip=None):
        clip = clip if clip is not None else self.clip

        if valid_mask is None:
            valid_mask = torch.ones_like(depth_linear).bool()
        valid_mask = valid_mask & (depth_linear > 0)

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

        # reverse all the operations done to normalize (only or reversible normalization)
        if (not self.is_absolute):
            logging.warning(f"{self.__class__} only for reversible normalization")
            return self.scale_back(depth_norm=depth_norm)
        else:
            depth = (depth_norm - self.norm_min) / self.norm_range * (self.max_depth - 0) + 0
            return depth
