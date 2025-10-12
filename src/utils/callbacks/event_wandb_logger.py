


import os
from typing import Any, Dict, Optional
from matplotlib import cm
import numpy as np
from datetime import datetime
import torch
from matplotlib.colors import Normalize
import torch.nn.functional as F
from torchvision.utils import make_grid
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.utilities import rank_zero_only
import wandb
from matplotlib import pyplot as plt
from lightning.pytorch.callbacks import RichProgressBar
from src.utils.callbacks.generic_wandb_logger import GenericLogger, hold_pbar
from src.utils.helper import cm_
from src.utils.metrics import calc_mae, calc_bias, calc_rmse, calc_abs_rel, calc_sq_rel, calc_rmse_log, calc_delta_acc, calc_mse
from torchvision.utils import save_image


class EventLogger(GenericLogger):
    def __init__(self, train_log_img_freq: int = 1000, train_log_score_freq: int = 1000,
                 train_ckpt_freq: int = 1000, show_samples_at_start: bool = False,
                 show_unconditional_samples: bool = False, check_freq_via: str = 'global_step',
                 enable_save_ckpt: bool = False, add_reference_artifact: bool = False,
                 report_sample_metrics: bool = True, sampling_batch_size: int = 6):
        """
        Callback to log images, scores and parameters to wandb.
        :param train_log_img_freq: frequency to log images. Set to -1 to disable
        :param train_log_score_freq: frequency to log scores. Set to -1 to disable
        :param train_ckpt_freq: frequency to log parameters. Set to -1 to disable
        :param show_samples_at_start: whether to log samples at the start of training (likely during sanity check)
        :param show_unconditional_samples: whether to log unconditional samples. Deprecated.
        :param check_freq_via: whether to check frequency via 'global_step' or 'epoch'
        :param enable_save_ckpt: whether to save checkpoint
        :param add_reference_artifact: whether to add the checkpoint as a reference artifact
        :param report_sample_metrics: whether to report sample metrics
        :param sampling_batch_size: number of samples to visualize
        """
        super().__init__(train_log_img_freq, train_log_score_freq, train_ckpt_freq, show_samples_at_start,
                         show_unconditional_samples, check_freq_via, enable_save_ckpt, add_reference_artifact,
                         report_sample_metrics, sampling_batch_size)
        # to be defined elsewhere
        self.depth_transformer = None
        self.resized_by_model = None

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        # set sampling batch size
        # if pl_module has a fucntion called set_sampling_batch_size, call it
        if hasattr(pl_module, 'set_sampling_batch_size'):
            pl_module.set_sampling_batch_size(self.sampling_batch_size)
        if hasattr(pl_module.hparams, 'allow_resize'):
            self.resized_by_model = True

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_fit_start(trainer, pl_module)
        self.depth_transformer = trainer.datamodule.depth_transform

    def log_score(self, pl_module: LightningModule, outputs: Dict[str, torch.Tensor]):
        pass
        # """
        # Called every train_log_score_freq steps to log per-batch metrics.
        # Now also applies resize logic if the model has resized its inputs.
        # """


    @hold_pbar("sampling...")
    def log_samples(self, trainer: Trainer, pl_module: LightningModule, outputs: Dict[str, torch.Tensor]):
        # pbar_taskid, original_pbar_desc = self._modify_pbar_desc(stage=trainer.state.stage)
        condition = outputs['condition']
        batch_dict = outputs['batch_dict']
        gt = batch_dict['depth_raw_norm']
        s = self.sampling_batch_size
        sample = pl_module.sample(condition[0:s])

        # 1) condition grid (first 3 channels)
        condition_grid = make_grid(condition[0:s].detach().cpu(), nrow=s, normalize=True)
        condition_grid = condition_grid.permute(1, 2, 0)[:, :, :3].numpy()

        # 2) pull out metrics (in meters)
        sample_metric = self.depth_transformer.denormalize(sample)
        gt_metric = self.depth_transformer.denormalize(gt[0:s])

        # 3) log‐invert → [0,1]
        eval_min = 5
        eval_max = 250
        sample_vis = map_depth_for_vis(sample_metric, amin=eval_min , amax=eval_max)  # torch [B,1,H,W]
        gt_vis = map_depth_for_vis(gt_metric, amin=eval_min , amax=eval_max)

        # 4) make the float grids
        sample_grid = make_grid(sample_vis, nrow=s, normalize=False)[0].detach().cpu().numpy()  # [H, W]
        gt_grid = make_grid(gt_vis, nrow=s, normalize=False)[0].detach().cpu().numpy()

        # 5) pick your meter ticks and their normalized locations
        # meter_ticks = torch.tensor([2.0, 5.0, 10.0, 20.0, 40.0, 80.0])
        meter_ticks = torch.tensor([5.0, 20.0, 80.0, 160.0, 250.0])
        tick_locs = map_depth_for_vis(meter_ticks, amin=eval_min , amax=eval_max).numpy()

        # 3) now build a 3×2 layout so col0 is images, col1 is colorbars:
        fig = plt.figure(figsize=(s * 7, 18))
        # slim down the colorbar column:
        gs = fig.add_gridspec(3, 2, width_ratios=[1, 0.01], wspace=0.01)
        ax0 = fig.add_subplot(gs[0, 0])
        # show RGB event condition, with gray=0, white=+1, black=-1
        im0 = ax0.imshow(condition_grid, cmap='gray', aspect='auto')
        ax0.set_title('Condition')
        ax0.axis('off')
        # colorbar for condition
        cax0 = fig.add_subplot(gs[0, 1])
        cbar0 = fig.colorbar(im0, cax=cax0)
        cbar0.set_label('Event polarity')
        cbar0.set_ticks([-1.0, 0.0, 1.0])
        cbar0.set_ticklabels(['-', '0', '+'])

        # --- Prediction row ---
        ax1 = fig.add_subplot(gs[1, 0])
        im1 = ax1.imshow(sample_grid, cmap='magma', vmin=0, vmax=1, aspect='auto')
        ax1.set_title('Prediction')
        ax1.axis('off')
        cax1 = fig.add_subplot(gs[1, 1])
        cbar1 = fig.colorbar(im1, cax=cax1)
        cbar1.set_label('Depth (m)')
        cbar1.set_ticks(tick_locs)
        cbar1.set_ticklabels([f"{int(m)}" for m in meter_ticks])

        # --- Ground Truth row ---
        ax2 = fig.add_subplot(gs[2, 0])
        im2 = ax2.imshow(gt_grid, cmap='magma', vmin=0, vmax=1, aspect='auto')
        ax2.set_title('Ground Truth')
        ax2.axis('off')
        cax2 = fig.add_subplot(gs[2, 1])
        cbar2 = fig.colorbar(im2, cax=cax2)
        cbar2.set_label('Depth (m)')
        cbar2.set_ticks(tick_locs)
        cbar2.set_ticklabels([f"{int(m)}" for m in meter_ticks])

        plt.tight_layout()
        wandb.log({
            'val/conditional_samples': wandb.Image(fig, caption='condition, prediction, GT')
        })
        plt.close(fig)

        # report metrics
        if self.report_sample_metrics:
            if self.resized_by_model:
                sample_metric_resized = F.interpolate(sample_metric, size=(gt_metric.shape[-2], gt_metric.shape[-1]),
                                                      mode='bilinear', align_corners=False)
            else:
                sample_metric_resized = sample_metric
            out_, gt_ = sample_metric_resized.cpu().detach().numpy(), gt_metric[0:s, :, :, :].cpu().detach().numpy()
            # calc metrics
            row = {'val_sample/mae': calc_mae(out_, gt_, k=1, pooling_func='mean'),
                   'val_sample/abs_rel': calc_abs_rel(out_, gt_, k=1, pooling_func='mean'),
                   'val_sample/sq_rel': calc_sq_rel(out_, gt_, k=1, pooling_func='mean'),
                   'val_sample/mse': calc_mse(out_, gt_, k=1, pooling_func='mean'),
                   'val_sample/_rmse_log': calc_rmse_log(out_, gt_, k=1, pooling_func='mean'),
                   'val_sample/delta_1.25': calc_delta_acc(out_, gt_, delta=1.25, k=1, pooling_func='mean'),
                   'val_sample/delta_1.25^2': calc_delta_acc(out_, gt_, delta=1.25 ** 2, k=1, pooling_func='mean'),
                   'val_sample/delta_1.25^3': calc_delta_acc(out_, gt_, delta=1.25 ** 3, k=1, pooling_func='mean'),
                   'val/sample_epoch': trainer.current_epoch,
                   }
            # wandb.log({'val/sample_mae': mae, 'val/sample_bias': bias, 'epoch': trainer.current_epoch})
            wandb.log(row)
        return

    def log_samples_helper(self, condition, sample, gt, n):
        fig, axs = plt.subplots(3, n, figsize=(n * 5, 15))
        for i in range(n):
            axs[0, i].imshow(condition[i, :, :, :].detach().cpu().numpy())
            axs[1, i].imshow(gt[i, 0, :, :].detach().cpu().numpy(), cmap='magma')
            axs[2, i].imshow(sample[i, 0, :, :].detach().cpu().numpy(), cmap='magma')

    from torchvision.utils import make_grid, save_image
    import matplotlib.pyplot as plt
    import os

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    # your datamodule must define `depth_transform`
        self.depth_transformer = trainer.datamodule.depth_transform

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):

        # only first batch
        if batch_idx != 0:
            return

        # 1) make a timestamped run dir
        ts = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = os.path.join("/home/qd8/eval", ts)
        os.makedirs(run_dir, exist_ok=True)

        # unpack
        condition = outputs["condition"]  # [B, C, H, W]
        gt_norm = outputs["batch_dict"]["depth_raw_norm"]  # [B, 1, H, W]
        B, C, H, W = condition.shape
        s = self.sampling_batch_size

        # get prediction (from outputs or by sampling)
        if outputs.get("prediction") is not None:
            pred_norm = outputs["prediction"]  # [B,1,H,W]
        else:
            pred_norm = pl_module.sample(condition[:s])  # [B,1,H,W]

        # 2) save per‐bin uint8 condition images for sample 0
        cond_dir = os.path.join(run_dir, "condition")
        os.makedirs(cond_dir, exist_ok=True)

        all_bins = []
        for i in range(B):
            arr = condition[i].detach().cpu().numpy()  # [C,H,W]
            vis = float_to_uint8_nbin(arr)  # [C,H,W]
            for b in range(C):
                all_bins.append(vis[b])  # each [H,W]
        # stack into tensor [B*C, 1, H, W]
        cond_vis_t = torch.from_numpy(np.stack(all_bins)).unsqueeze(1)
        # grid: C columns, B rows
        cond_grid = make_grid(cond_vis_t, nrow=C, normalize=False)[0]  # [H*B, W*C]
        plt.imsave(
            os.path.join(cond_dir, "all_bins_grid.png"),
            cond_grid.numpy(),
            cmap="gray",
            vmin=0,
            vmax=255
        )

        # 3) denormalize & clamp to [5,250]m, then map to [0,1]
        amin_vis, amax_vis = 2.0, 80.0
        raw_gt = self.depth_transformer.denormalize(gt_norm)[:, 0]
        raw_gt = torch.nan_to_num(raw_gt,
                                  nan=amax_vis, posinf=amax_vis, neginf=amin_vis)
        raw_pred = self.depth_transformer.denormalize(pred_norm)[:, 0]

        vis_gt = map_depth_for_vis(raw_gt, amin=amin_vis, amax=amax_vis)  # [B,H,W]
        vis_pred = map_depth_for_vis(raw_pred, amin=amin_vis, amax=amax_vis)

        # 4) compute metrics on sample 0
        # valid0 = torch.isfinite(raw_gt[0])
        # p0, g0 = raw_pred[0][valid0], raw_gt[0][valid0]
        # if p0.numel() > 0:
        #     mae0 = (p0 - g0).abs().mean().item()
        #     mse0 = ((p0 - g0) ** 2).mean().item()
        #     rmse0 = mse0 ** 0.5
        # else:
        #     mae0 = mse0 = rmse0 = float('nan')
        # print(f"[TestVisCallback] Sample0 → MAE={mae0:.3f}m, RMSE={rmse0:.3f}m")

        # 5) build 2×B grid: top=Prediction, bottom=GT, with shared colorbar
        comp_dir = os.path.join(run_dir, "comparison")
        os.makedirs(comp_dir, exist_ok=True)

        # 5) build comparison figure: 2 rows, B image columns + 1 colorbar column
        comp_dir = os.path.join(run_dir, "comparison")
        os.makedirs(comp_dir, exist_ok=True)

        B_show = min(B, s)

        # grids (no black borders)
        pred_grid = make_grid(
            vis_pred[:B_show].unsqueeze(1), nrow=B_show, normalize=False
        )[0].cpu().numpy()
        gt_grid = make_grid(
            vis_gt[:B_show].unsqueeze(1), nrow=B_show, normalize=False
        )[0].cpu().numpy()

        # figure: 2 rows (Pred, GT) × 2 cols (image, thin colorbar)
        fig = plt.figure(figsize=(B_show * 7, 12))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.015], wspace=0.02, hspace=0.08)

        # meter ticks (same as validation)
        meter_ticks = torch.tensor([2.0, 5.0, 10.0, 20.0, 40.0, 80.0])
        tick_locs = map_depth_for_vis(meter_ticks, amin=amin_vis, amax=amax_vis).numpy()
        tick_labels = [f"{int(m)}" for m in meter_ticks]

        # Row 0: Prediction + its own colorbar (same normalization)
        ax0 = fig.add_subplot(gs[0, 0])
        im0 = ax0.imshow(pred_grid, cmap="magma", vmin=0, vmax=1, aspect="auto")
        ax0.set_title("Prediction");
        ax0.axis("off")
        cax0 = fig.add_subplot(gs[0, 1])
        cb0 = fig.colorbar(im0, cax=cax0, orientation="vertical")
        cb0.set_label("Depth (m)")
        cb0.set_ticks(tick_locs);
        cb0.set_ticklabels(tick_labels)

        # Row 1: Ground Truth + its own colorbar (same normalization)
        ax1 = fig.add_subplot(gs[1, 0])
        im1 = ax1.imshow(gt_grid, cmap="magma", vmin=0, vmax=1, aspect="auto")
        ax1.set_title("Ground Truth");
        ax1.axis("off")
        cax1 = fig.add_subplot(gs[1, 1])
        cb1 = fig.colorbar(im1, cax=cax1, orientation="vertical")
        cb1.set_label("Depth (m)")
        cb1.set_ticks(tick_locs);
        cb1.set_ticklabels(tick_labels)

        plt.tight_layout()
        fig.savefig(os.path.join(comp_dir, "batch0_pred_gt_valstyle.png"),
                    bbox_inches="tight", pad_inches=0)
        plt.close(fig)


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
            # only display 1st 3 channels
            if grid.shape[0] > 3:
                grid = grid[:3, :, :]
            if 'depth_raw_norm' == k:
                grid_mono = self.depth_transformer.denormalize(grid_mono)
            if 'depth' in k:
                grid_mono = map_depth_for_vis(grid_mono, amin=5, amax=250)
            if vis_param[k] is None:
                images = wandb.Image(grid, caption=k)
            else:
                cm_grid = cm_(grid_mono.detach().cpu(), vis_param[k])
                images = wandb.Image(cm_grid, caption=k)
            wandb.log({f"dataloader/{k}": images})
        return


############################ STATIC METHODS ############################

def map_depth_for_vis(img, amax, amin=None):
    if amin is None:
        amin = 5.0
    img = torch.as_tensor(img)

    # mark invalid: zeros, negatives, NaNs, ±inf
    invalid = (img <= 0)

    d = torch.clip(img, min=float(amin), max=float(amax))
    v = torch.log(torch.as_tensor(amax, device=d.device, dtype=d.dtype) / d)
    denom = max(np.log(float(amax) / float(amin)), 1e-12)
    out = v / denom
    # out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    # force invalids to 0 in the visualization
    out[invalid] = 0.0
    return out




def float_to_uint8_nbin(arr: np.ndarray) -> np.ndarray:
    """
    Linearly scale float NBIN array to uint8 [0,255] per-bin independently.
    """
    out = np.empty_like(arr, dtype=np.uint8)
    for b in range(arr.shape[0]):
        channel = arr[b]
        vmin, vmax = float(channel.min()), float(channel.max())
        if vmax <= vmin:
            out[b] = np.zeros_like(channel, dtype=np.uint8)
        else:
            scaled = (channel - vmin) / (vmax - vmin)
            scaled = (scaled * 255.0).round().clip(0, 255).astype(np.uint8)
            out[b] = scaled
    return out