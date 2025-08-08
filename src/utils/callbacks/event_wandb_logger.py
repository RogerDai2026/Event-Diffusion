


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
        sample_vis = map_depth_for_vis(sample_metric, amin=5, amax=250)  # torch [B,1,H,W]
        gt_vis = map_depth_for_vis(gt_metric, amin=5, amax=250)

        # 4) make the float grids
        sample_grid = make_grid(sample_vis, nrow=s, normalize=True)[0].detach().cpu().numpy()  # [H, W]
        gt_grid = make_grid(gt_vis, nrow=s, normalize=True)[0].detach().cpu().numpy()

        # 5) pick your meter ticks and their normalized locations
        meter_ticks = torch.tensor([5.0, 10.0, 20.0, 50.0, 100.0, 250.0])
        tick_locs = map_depth_for_vis(meter_ticks, amax=250.0, amin=5.0).numpy()

        # 3) now build a 3×2 layout so col0 is images, col1 is colorbars:
        fig = plt.figure(figsize=(s * 7, 18))
        # slim down the colorbar column:
        gs = fig.add_gridspec(3, 2, width_ratios=[1, 0.01], wspace=0.01)
        ax0 = fig.add_subplot(gs[0, 0])
        # show RGB event condition, with gray=0, white=+1, black=-1
        im0 = ax0.imshow(condition_grid, cmap='gray', vmin=-1, vmax=1, aspect='auto')
        ax0.set_title('Condition')
        ax0.axis('off')
        # colorbar for condition
        cax0 = fig.add_subplot(gs[0, 1])
        cbar0 = fig.colorbar(im0, cax=cax0)
        cbar0.set_label('Event polarity')
        cbar0.set_ticks([-1.0, 0.0, 1.0])
        cbar0.set_ticklabels(['-', '0', '+'])
        cbar0.set_label('Normalized value')

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
        amin_vis, amax_vis = 5.0, 150.0
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

        # create figure with (2 rows x (B+1) columns)
        fig = plt.figure(figsize=(4 * (B + 1), 8))
        gs = fig.add_gridspec(
            2, B + 1,
            width_ratios=[*([1] * B), 0.05],
            wspace=0.1, hspace=0.15
        )

        # meter ticks
        meter_ticks = torch.tensor([5.0, 10.0, 20.0, 50.0, 100.0, 150])
        tick_locs = map_depth_for_vis(meter_ticks, amin=amin_vis, amax=amax_vis).numpy()

        # row 0: Predictions
        for i in range(B):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(vis_pred[i].cpu().numpy(), cmap="magma", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel("Prediction", fontsize=12)

        cax0 = fig.add_subplot(gs[0, B])
        cbar0 = fig.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="magma"),
            cax=cax0, orientation="vertical"
        )
        cbar0.set_ticks(tick_locs)
        cbar0.set_ticklabels([f"{int(m)}m" for m in meter_ticks])
        cbar0.set_label("Depth (m)", rotation=90, labelpad=8)

        # row 1: Ground Truth
        for i in range(B):
            ax = fig.add_subplot(gs[1, i])
            ax.imshow(vis_gt[i].cpu().numpy(), cmap="magma", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel("Ground Truth", fontsize=12)

        cax1 = fig.add_subplot(gs[1, B])
        cbar1 = fig.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="magma"),
            cax=cax1, orientation="vertical"
        )
        cbar1.set_ticks(tick_locs)
        cbar1.set_ticklabels([f"{int(m)}m" for m in meter_ticks])
        cbar1.set_label("Depth (m)", rotation=90, labelpad=8)

        # save and close
        fig.savefig(
            os.path.join(comp_dir, "batch0_pred_vs_gt.png"),
            bbox_inches="tight", pad_inches=0
        )
        plt.close(fig)

        print(f"[TestVisCallback] Saved condition grid and comparison under {run_dir}")

        # # --- save prediction and ground truth separately with shared colorbar ---
        # # denormalize
        # pred_metric = self.depth_transformer.denormalize(sample[0:1])[0, 0].detach().cpu().numpy()
        # pred_metric =  map_depth_for_vis(pred_metric, amin=5, amax = 80)
        #
        # gt_metric = self.depth_transformer.denormalize(gt[0:1])[0, 0].detach().cpu().numpy()
        # gt_metric =  map_depth_for_vis(gt_metric, amin=5, amax = 80)
        #
        # # use fixed range or dynamic across both
        # amin, amax = 2, 80
        #
        # norm = Normalize(vmin=amin, vmax=amax)
        # cmap = plt.get_cmap('magma')
        #
        # # prediction
        # fig, ax = plt.subplots(figsize=(6, 5))
        # im = ax.imshow(pred_metric, cmap=cmap, norm=norm)
        # ax.set_title('Prediction')
        # ax.axis('off')
        # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # fig.savefig(os.path.join(save_dir, 'prediction.png'), bbox_inches='tight', pad_inches=0)
        # plt.close(fig)
        #
        # # ground truth
        # fig, ax = plt.subplots(figsize=(6, 5))
        # im = ax.imshow(gt_metric, cmap=cmap, norm=norm)
        # ax.set_title('Ground Truth')
        # ax.axis('off')
        # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # fig.savefig(os.path.join(save_dir, 'ground_truth.png'), bbox_inches='tight', pad_inches=0)
        # plt.close(fig)
        #
        # print(f"[TestVisCallback] Saved input_bins.png, prediction.png, ground_truth.png to {save_dir}")

        # save_dir = "/home/qd8/eval"
        # os.makedirs(save_dir, exist_ok=True)
        #
        # condition = outputs["condition"]  # [B, C, H, W]
        # batch_dict = outputs["batch_dict"]
        # gt = batch_dict["depth_raw_norm"]  # [B,1,H,W]
        # s = self.sampling_batch_size
        #
        # if "prediction" in outputs and outputs["prediction"] is not None:
        #     sample = outputs["prediction"]
        # else:
        #     sample = pl_module.sample(condition[0:s])  # [s, ...] prediction
        #
        # # ---- condition: only first 3 bins as RGB ----
        # cond_subset = condition[0:s]
        # C = cond_subset.shape[1]
        # if C >= 3:
        #     cond_rgb = cond_subset[:, :3]  # take first three bins
        # elif C == 1:
        #     cond_rgb = cond_subset.repeat(1, 3, 1, 1)
        # else:
        #     # pad to 3 channels if 2
        #     pad = torch.zeros_like(cond_subset[:, : (3 - C), :, :])
        #     cond_rgb = torch.cat([cond_subset, pad], dim=1)
        # condition_grid = make_grid(cond_rgb.detach().cpu(), nrow=s, normalize=True)
        # # convert to HWC
        # condition_grid = condition_grid.permute(1, 2, 0).numpy()
        #
        # # ---- sample / prediction ----
        # sample_metric = self.depth_transformer.denormalize(sample[0:s])
        # sample_vis = map_depth_for_vis(sample_metric, amin=5, amax=25000)
        # sample_grid = make_grid(sample_vis, nrow=s, normalize=True)[0, :, :]
        # sample_grid_cm = cm_(sample_grid.detach().cpu(), "magma")
        # if sample_grid_cm.shape[-1] == 4:
        #     sample_grid_cm = sample_grid_cm[..., :3]
        #
        # # ---- ground truth ----
        # gt_metric = self.depth_transformer.denormalize(gt[0:s])
        # gt_grid= make_grid(gt_metric, nrow=s, normalize=True)
        # gt_grid_np = gt_grid.permute(1, 2, 0).numpy()
        # # [H, s*W]
        # # gt_grid_cm = cm_(gt_grid.detach().cpu(), "magma")
        # # if sample_grid_cm.shape[-1] == 4:
        # #     sample_grid_cm = sample_grid_cm[..., :3]
        #
        # # put 3 grids in one plt image, row by row
        # fig, axs = plt.subplots(3, 1, figsize=(s * 5, 15))
        # axs[0].imshow(condition_grid)
        # axs[1].imshow(sample_grid_cm)
        # axs[2].imshow(gt_grid_np)
        # for ax in axs:
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        # plt.tight_layout()
        #
        # out_path = os.path.join(save_dir, f"vis_1.png")
        # fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        # plt.close(fig)
        # print(f"[TestVisCallback] Saved visualization to {out_path}")

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