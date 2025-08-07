


import os
from typing import Any, Dict, Optional
from matplotlib import cm
import numpy as np
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
        gt = outputs['batch_dict']['depth_raw_norm']
        # config = pl_module.model_config
        # s = pl_module.model_config.sampling.sampling_batch_size
        s = self.sampling_batch_size
        sample = pl_module.sample(condition[0:s])

        condition_grid = make_grid(condition[0:s].detach().cpu(), nrow=s, normalize=True).permute(1, 2, 0)
        # only display 1st 3 channels
        condition_grid = condition_grid[:, :, 0:3]

        sample_metric = self.depth_transformer.denormalize(sample)
        sample_vis = map_depth_for_vis(sample_metric, amin=5, amax=250)
        sample_grid = make_grid(sample_vis, nrow=s, normalize=True)[0, :, :]
        sample_grid_cm = cm_(sample_grid.detach().cpu(), 'magma')

        gt_metric = self.depth_transformer.denormalize(gt[0:s])
        gt_vis = map_depth_for_vis(gt_metric, amin=5, amax=250)
        gt_grid = make_grid(gt_vis, nrow=s, normalize=True)[0, :, :]
        gt_grid_cm = cm_(gt_grid.detach().cpu(), 'magma')

        # put 3 grids in one plt image, row by row
        fig, axs = plt.subplots(3, 1, figsize=(s*5, 15))
        axs[0].imshow(condition_grid.detach().cpu().numpy())
        axs[1].imshow(sample_grid_cm)
        axs[2].imshow(gt_grid_cm)
        # turn off ticks
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.close(fig)
        images = wandb.Image(fig, caption="condition, prediction, GT")
        wandb.log({"val/conditional_samples": images})

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
        save_dir = "/home/qd8/eval"
        os.makedirs(save_dir, exist_ok=True)

        condition = outputs["condition"]  # [B, C, H, W]
        gt = outputs["batch_dict"]["depth_raw_norm"]  # [B,1,H,W]
        s = self.sampling_batch_size

        # prediction
        if "prediction" in outputs and outputs["prediction"] is not None:
            sample = outputs["prediction"]
        else:
            sample = pl_module.sample(condition[0:s])

        # --- visualize per-bin input channels ---
        img = condition[0].detach().cpu().numpy()  # [C,H,W]
        vis = float_to_uint8_nbin(img)

        # print stats
        for b in range(img.shape[0]):
            channel = img[b]
            nz = np.count_nonzero(channel)
            total = channel.size
            unique = np.unique(channel)
            print(
                f"[float] Bin {b}: min={channel.min():.4f}, max={channel.max():.4f}, sum={channel.sum():.4f}, nonzero={nz}/{total}, unique_sample={unique[:5]}{'...' if unique.size > 5 else ''}")
            if nz > 0:
                coords = np.argwhere(channel != 0)
                sample_vals = [float(channel[tuple(c)]) for c in coords[:5]]
                print(f"    sample nonzero coords: {coords[:5].tolist()[:5]} values: {sample_vals}")
            uchannel = vis[b]
            unique_u = np.unique(uchannel)
            print(
                f"[uint8] Bin {b}: min={int(uchannel.min())}, max={int(uchannel.max())}, unique_sample={unique_u[:5]}{'...' if unique_u.size > 5 else ''}")
        if np.count_nonzero(img) == 0:
            print("WARNING: ENTIRE IMAGE IS ZERO")

        # save input bins
        fig, axs = plt.subplots(vis.shape[0], 1, figsize=(6, 2 * vis.shape[0]))
        for b in range(vis.shape[0]):
            ax = axs[b] if vis.shape[0] > 1 else axs
            im = ax.imshow(vis[b], cmap="gray", vmin=0, vmax=255, aspect="auto")
            ax.set_title(f"Bin {b}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.01)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "input_bins.png"), bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        # … your code that denormalizes to raw_pred / raw_gt, but keep them as torch.Tensor …
        raw_pred = self.depth_transformer.denormalize(sample[0:1])[0, 0].detach().cpu()
        raw_gt = self.depth_transformer.denormalize(gt[0:1])[0, 0].detach().cpu()# -Inf → amin_vis

        # display‐range in meters
        amin_vis, amax_vis = 5.0, 100.0

        raw_gt = torch.nan_to_num(raw_gt,
                                  nan=amax_vis,  # clamp NaNs to amax_vis
                                  posinf=amax_vis,  # +Inf → amax_vis
                                  neginf=amin_vis)

        # 1) apply your log‐invert & normalize helper *on the tensor*:
        vis_pred = map_depth_for_vis(raw_pred, amax=amax_vis, amin=amin_vis)
        vis_gt = map_depth_for_vis(raw_gt, amax=amax_vis, amin=amin_vis)

        # 2) Build a valid‐mask: finite & non‐NaN & non‐Inf
        valid_mask = torch.isfinite(raw_gt)

        # 3) Optionally also exclude zeros or out‐of‐range:
        # valid_mask &= (raw_gt > 0.0)

        # 4) Gather only valid pixels
        pred_vals = raw_pred[valid_mask]
        gt_vals = raw_gt[valid_mask]

        # 5) Compute metrics
        if pred_vals.numel() > 0:
            mae = torch.mean(torch.abs(pred_vals - gt_vals)).item()
            mse = torch.mean((pred_vals - gt_vals) ** 2).item()
            abs_err = torch.abs(pred_vals - gt_vals)  # per-pixel if you need the full map
        else:
            mae = mse = float('nan')

        print(f"[TestVisCallback]   MAE over valid pixels = {mae:.4f} m")
        print(f"[TestVisCallback]   MSE over valid pixels = {mse:.4f} (m²)")
        # if you want RMSE:
        print(f"[TestVisCallback]   RMSE = {mse ** 0.5:.4f} m")

        for name, vis_img in [("Prediction", vis_pred), ("Ground Truth", vis_gt)]:
            fig, ax = plt.subplots(figsize=(6, 5))
            # 2) plot the already‐normalized image with vmin/vmax=(0,1)
            im = ax.imshow(vis_img.numpy(), cmap="magma", vmin=0, vmax=1)
            ax.set_title(name)
            ax.axis("off")

            # 3) build a colorbar whose tick‐labels are in meters
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Depth (m)")

            # pick the real‐world depths you want labeled
            # meter_ticks = torch.tensor([5.0, 20.0, 40.0, 80.0, 160.0, 500.0, 1000.0])
            meter_ticks = torch.tensor([5.0, 10.0, 20.0, 40.0, 80.0, 100.0])

            # remap them into [0,1] via the same helper
            tick_positions = map_depth_for_vis(meter_ticks, amax=amax_vis, amin=amin_vis).numpy()

            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels([f"{int(m)}" for m in meter_ticks])

            fig.savefig(os.path.join(save_dir, f"{name.replace(' ', '_').lower()}.png"),
                        bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        print(f"[TestVisCallback] wrote enhanced log‐invert plots to {save_dir}")
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