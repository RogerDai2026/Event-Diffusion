# src/utils/callbacks/vae_logger.py

import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import make_grid
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from src.utils.callbacks.generic_wandb_logger import GenericLogger, hold_pbar
from src.utils.metrics import calc_mae, calc_bias, calc_rmse, calc_abs_rel, calc_sq_rel, calc_rmse_log, calc_delta_acc, calc_mse
from lightning import LightningModule, Trainer
import os
from torchvision.utils import save_image

class VAELogger(GenericLogger):
    """
    Logs VAE inputs vs. reconstructions (first 3 channels) each validation epoch.
    """

    def __init__(self,
                 train_log_img_freq: int = 1,
                 train_log_score_freq: int = -1,
                 train_ckpt_freq: int = -1,
                 show_samples_at_start: bool = False,
                 sampling_batch_size: int = 8,
                 check_freq_via: str = 'epoch',
                 enable_save_ckpt: bool = True):
        # We only care about val‐time visualization
        super().__init__(
            train_log_img_freq=train_log_img_freq,
            train_log_score_freq=train_log_score_freq,
            train_ckpt_freq=train_ckpt_freq,
            show_samples_at_start=show_samples_at_start,
            sampling_batch_size=sampling_batch_size,
            check_freq_via = check_freq_via,
            enable_save_ckpt = enable_save_ckpt
        )
    #     self.depth_transformer = None
    #
    #
    # def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #     super().on_fit_start(trainer, pl_module)
    #     self.depth_transformer = trainer.datamodule.depth_transform


    def log_score(self, pl_module, outputs):
        # Not used for VAE
        pass

    def visualize_batch(self, **batch):
        # Called once at train start if show_samples_at_start=True
        x = batch.get("input", None)
        if x is None:
            return
        k = min(4, x.shape[0])
        # take only first 3 channels for display
        vis = x[:k, :3]
        grid = make_grid(vis, nrow=k, normalize=True)
        wandb.log({"train/vae_inputs": wandb.Image(grid)})

    def _event_to_disp(x_bin: torch.Tensor) -> np.ndarray:
        # baseline is exactly the original 128→normed value
        baseline = (128 / 255.0) * 2.0 - 1.0
        disp = torch.zeros_like(x_bin)
        above = x_bin > baseline
        below = x_bin < baseline
        # scale positives to (0,1]
        disp[above] = (x_bin[above] - baseline) / (1.0 - baseline)
        # scale negatives to [–1,0)
        disp[below] = (x_bin[below] - baseline) / (baseline + 1.0)
        return disp.cpu().numpy()

    @hold_pbar("Visualizing VAE reconstructions…")
    @rank_zero_only
    def log_samples(self, trainer, pl_module, outputs):
        # ensure outputs is a list
        if isinstance(outputs, dict):
            outputs = [outputs]

        condition = torch.cat([o["input"] for o in outputs], dim=0)  # [N,5,H,W]

        # 2) reconstruct all N samples
        with torch.no_grad():
            recon_all = pl_module.sample(condition)  # [N,5,H,W]

            # full-set metrics
            mse = F.mse_loss(recon_all, condition).item()
            mae = F.l1_loss(recon_all, condition).item()
            # dynamic range for PSNR
            dr = (condition.max() - condition.min()).clamp_min(1e-12)
            psnr = (10 * torch.log10(dr ** 2 / (mse + 1e-12))).item()

            gt_mask = (condition > 0.0)  # True wherever there was an event
            pred_mask = (recon_all > 0.0)  # True wherever you predicted one

            recon_rate = (gt_mask == pred_mask).float().mean().item()

            # log scalars once for the whole epoch
        wandb.log({
                "val_sample/mse": mse,
                "val_sample/mae": mae,
                "val_sample/psnr": psnr,
                "val_sample/reconstruction_rate": recon_rate,
            })

        # 3) visualize just the first s samples
        s = self.sampling_batch_size
        cond_s = condition[:s]
        rec_s = recon_all[:s]

        cond_grid = make_grid(cond_s.cpu(), nrow=s, normalize=True).permute(1, 2, 0)[:, :, :3]
        rec_grid = make_grid(rec_s.cpu(), nrow=s, normalize=True).permute(1, 2, 0)[:, :, :3]

        fig, axs = plt.subplots(2, 1, figsize=(s * 5, 15))
        axs[0].imshow(cond_grid.numpy());
        axs[0].set_title("Input (bins 0–2)")
        axs[1].imshow(rec_grid.numpy());
        axs[1].set_title("Recon (bins 0–2)")
        for ax in axs:
            ax.set_xticks([]);
            ax.set_yticks([])
        plt.tight_layout()
        plt.close(fig)

        wandb.log({"val/reconstruction": wandb.Image(fig)})
        # B, C, H, W = sample.shape
        # k = min(self.sampling_batch_size, B)
        #
        # # compute per-sample metrics (optional) …
        # # … your MSE/SSIM code here …
        #
        # print("x shape:", inputs.shape)  # -> torch.Size([2, 5, 256, 352])
        # print("recon shape:", recons.shape)
        #
        # inp_vis = inputs[:k, :3]  # [k,3,H,W]
        # rec_vis = recons[:k, :3]
        #
        # inp_grid = make_grid(inp_vis, nrow=k, normalize=False)
        # rec_grid = make_grid(rec_vis, nrow=k, normalize=False)
        #
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(k * 3, 6))
        # ax1.imshow(inp_grid.permute(1, 2, 0).cpu().numpy())
        # ax1.set_title("VAE Inputs (channels 0–2)")
        # ax1.axis("off")
        # ax2.imshow(rec_grid.permute(1, 2, 0).cpu().numpy())
        # ax2.set_title("VAE Reconstructions")
        # ax2.axis("off")
        # plt.tight_layout()
        # plt.close(fig)
        print(f"Logging VAE at epoch {trainer.current_epoch}")
        # wandb.log({"val/vae_recon_grid": wandb.Image(fig)})

    @rank_zero_only
    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0
    ):
        """Save input vs reconstruction after each test batch."""
        # where to write
        save_dir = "/home/qd8/eval/vae"
        os.makedirs(save_dir, exist_ok=True)

        if outputs is None:
            return

        # Expect your test_step to return {"input": x, "recon": recon}
        x     = outputs.get("input")
        recon = outputs.get("recon")

        if x is None or recon is None:
            return

        B = x.shape[0]
        n = min(B, 8)
        for i in range(n):
            # inputs (first 3 bins as RGB)
            inp = x[i]
            if inp.ndim == 3 and inp.shape[0] > 3:
                inp = inp[:3]
            save_image(
                inp,
                os.path.join(save_dir, f"batch{batch_idx:03d}_sample{i:02d}_input.png"),  normalize=True,
            )

            # reconstructions (first 3 bins as RGB)
            out = recon[i]
            if out.ndim == 3 and out.shape[0] > 3:
                out = out[:3]
            save_image(
                out,
                os.path.join(save_dir, f"batch{batch_idx:03d}_sample{i:02d}_recon.png"), normalize=True,
            )

        print(f"[VAELogger] saved {n} inputs+recons for test batch {batch_idx} → {save_dir}")