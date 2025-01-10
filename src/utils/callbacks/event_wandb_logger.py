import os
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.utilities import rank_zero_only
import wandb
from matplotlib import pyplot as plt
from lightning.pytorch.callbacks import RichProgressBar
from src.utils.callbacks.generic_wandb_logger import GenericLogger
from src.utils.helper import cm_


class EventLogger(GenericLogger):
    def __init__(self, train_log_img_freq: int = 1000, train_log_score_freq: int = 1000,
                 train_log_param_freq: int = 1000, show_samples_at_start: bool = False,
                 show_unconditional_samples: bool = False, check_freq_via: str = 'global_step',
                 enable_save_ckpt: bool = False, add_reference_artifact: bool = False,
                 report_sample_metrics: bool = True):
        """
        Callback to log images, scores and parameters to wandb.
        :param train_log_img_freq: frequency to log images. Set to -1 to disable
        :param train_log_score_freq: frequency to log scores. Set to -1 to disable
        :param train_log_param_freq: frequency to log parameters. Set to -1 to disable
        :param show_samples_at_start: whether to log samples at the start of training (likely during sanity check)
        :param show_unconditional_samples: whether to log unconditional samples. Deprecated.
        :param check_freq_via: whether to check frequency via 'global_step' or 'epoch'
        :param enable_save_ckpt: whether to save checkpoint
        :param add_reference_artifact: whether to add the checkpoint as a reference artifact
        :param report_sample_metrics: whether to report sample metrics
        """
        super().__init__(train_log_img_freq, train_log_score_freq, train_log_param_freq, show_samples_at_start,
                         show_unconditional_samples, check_freq_via, enable_save_ckpt, add_reference_artifact,
                         report_sample_metrics)

    def log_score(self, pl_module: LightningModule, outputs: Dict[str, torch.Tensor]):
        pass # TODO

    def log_samples(self, trainer: Trainer, pl_module: LightningModule, outputs: Dict[str, torch.Tensor]):
        pass # TODO



############################ STATIC METHODS ############################

def visualize_batch(**kwargs):
    vis_param = {
        'rgb_int': None,
        'rgb_norm': None,
        'depth_raw_linear': 'inferno',
        'depth_filled_linear': 'inferno',
        'valid_mask_raw': 'inferno',
        'valid_mask_filled': 'inferno',
        'depth_raw_norm': 'inferno',
        'depth_filled_norm': 'inferno',
    }
    n = 5
    for k, v in kwargs.items():
        grid = make_grid(v[0:n, :, :, :])
        grid_mono = grid[0, :, :].unsqueeze(0)
        if vis_param[k] is None:
            images = wandb.Image(grid, caption=k)
        else:
            cm_grid = cm_(grid_mono.detach().cpu(), vis_param[k])
            images = wandb.Image(cm_grid, caption=k)
        wandb.log({f"dataloader/{k}": images})
    return