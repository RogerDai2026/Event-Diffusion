import os
from typing import Any, Dict, Optional
import numpy as np
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
from src.utils.metrics import calc_mae, calc_bias


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
        # to be defined elsewhere
        self.depth_transformer = None

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_fit_start(trainer, pl_module)
        self.depth_transformer = trainer.datamodule.depth_transform

    def log_score(self, pl_module: LightningModule, outputs: Dict[str, torch.Tensor]):
       pass # TODO

    def log_samples(self, trainer: Trainer, pl_module: LightningModule, outputs: Dict[str, torch.Tensor]):
        pbar_taskid, original_pbar_desc = self._modify_pbar_desc()

        condition = outputs['condition']
        batch_dict = outputs['batch_dict']
        gt = outputs['batch_dict']['depth_raw_norm']
        config = pl_module.model_config
        s = pl_module.model_config.sampling.sampling_batch_size
        # sample = pl_module.sample(condition[0:s])

        # save this tensor
        sample = torch.load('/home/yl241/workspace/Event-WassDiff/tmp/sample.pt')
        # torch.save(sample, '/home/yl241/workspace/Event-WassDiff/tmp/sample.pt')
        print('sample saved')

        condition_grid = make_grid(condition[0:s].detach().cpu().permute(1, 2, 0), nrow=1)

        sample_metric = self.depth_transformer.denormalize(sample)
        sample_vis = map_depth_for_vis(sample_metric, amin=5, amax=25000)
        sample_grid = make_grid(sample_vis, nrow=1)[0, :, :].unsqueeze(0)
        sample_grid_cm = cm_(sample_grid.detach().cpu().permute(1, 2, 0), 'magma')

        gt_metric = self.depth_transformer.denormalize(gt[0:s])
        gt_vis = map_depth_for_vis(gt_metric, amin=5, amax=25000)
        gt_grid = make_grid(gt_vis, nrow=1)[0, :, :].unsqueeze(0)
        gt_grid_cm = cm_(gt_grid.detach().cpu().permute(1, 2, 0), 'magma')

        # put 3 grids in one plt image, row by row
        fig, axs = plt.subplots(3, 1, figsize=(5, 15))
        axs[0].imshow(condition_grid.detach().cpu().numpy())
        axs[1].imshow(gt_grid_cm.detach().cpu().numpy())
        axs[2].imshow(sample_grid_cm.detach().cpu().numpy())

        plt.tight_layout()
        plt.close(fig)
        images = wandb.Image(fig)
        wandb.log({"val/conditional_samples": images})

        # report metrics
        # if self.report_sample_metrics:
        #     mae = calc_mae(sample_metric.cpu().detach().numpy(), gt[0:s, :, :, :].cpu().detach().numpy(), valid_mask=None, k=1,
        #                    pooling_func='mean')
        return

    def log_samples_helper(self, condition, sample, gt, n):
        fig, axs = plt.subplots(3, n, figsize=(n * 5, 15))
        for i in range(n):
            axs[0, i].imshow(condition[i, :, :, :].detach().cpu().numpy())
            axs[1, i].imshow(gt[i, 0, :, :].detach().cpu().numpy(), cmap='magma')
            axs[2, i].imshow(sample[i, 0, :, :].detach().cpu().numpy(), cmap='magma')


    def visualize_batch(self, **kwargs):
        vis_param = {
            # 'rgb_int': None,
            'rgb_norm': None,
            'depth_raw_linear': 'magma',
            # 'depth_filled_linear': 'magma',
            # 'valid_mask_raw': None,
            # 'valid_mask_filled': None,
            'depth_raw_norm': 'magma',
            # 'depth_filled_norm': 'magma',
        }
        allowed_entries = vis_param.keys()
        n = 5
        for k, v in kwargs.items():
            if k not in allowed_entries:
                continue
            grid = make_grid(v[0:n, :, :, :])
            grid_mono = grid[0, :, :].unsqueeze(0)
            if 'depth_raw_norm' == k:
                grid_mono = self.depth_transformer.denormalize(grid_mono)
            if 'depth' in k:
                grid_mono = map_depth_for_vis(grid_mono, amin=5, amax=25000)
            if vis_param[k] is None:
                images = wandb.Image(grid, caption=k)
            else:
                cm_grid = cm_(grid_mono.detach().cpu(), vis_param[k])
                images = wandb.Image(cm_grid, caption=k)
            wandb.log({f"dataloader/{k}": images})
        return


############################ STATIC METHODS ############################

def map_depth_for_vis(img, amax, amin=None):
    img_temp = torch.clip(img, min=amin, max=amax)
    img_temp = torch.log(1/img_temp)
    img_temp = (img_temp - img_temp.min())/torch.abs(img_temp.max() - img_temp.min())
    return img_temp