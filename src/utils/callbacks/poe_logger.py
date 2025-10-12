# src/utils/callbacks/poe_logger.py
import os
import torch
import wandb
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from src.utils.callbacks.generic_wandb_logger import GenericLogger, hold_pbar


def map_depth_for_vis(img, amax, amin=None, valid_mask=None):
    """
    Map depth values to [0,1] range for visualization using log-inverse mapping.
    Same function as in event_wandb_logger.py for consistent depth visualization.
    
    Args:
        img: Depth tensor
        amax: Maximum depth value
        amin: Minimum depth value  
        valid_mask: Optional mask indicating valid pixels (True=valid, False=invalid)
    """
    import numpy as np
    
    # if amin is None:
    #     amin = 5.0
    img = torch.as_tensor(img)
    #
    # # mark invalid: zeros, negatives, NaNs, ±inf
    invalid = (img <= 0)

    d = torch.clip(img, min=float(amin), max=float(amax))
    v = torch.log(torch.as_tensor(amax, device=d.device, dtype=d.dtype) / d)
    denom = max(np.log(float(amax) / float(amin)), 1e-12)
    out = v / denom
    
    # Apply valid_mask if provided: invalid pixels → 0.0 (black in magma)
    if valid_mask is not None:
        # Ensure mask has same spatial dimensions
        if valid_mask.ndim == 4 and out.ndim == 4:
            if valid_mask.size(1) != out.size(1):
                valid_mask = valid_mask.expand_as(out)
        elif valid_mask.ndim == 3 and out.ndim == 4:
            # Broadcast mask to match batch and channel dimensions
            valid_mask = valid_mask.unsqueeze(1).expand_as(out)
        
        # Set invalid pixels to 0.0 (black in visualization)
        out = out * valid_mask.float()
    else:
        # Fallback: set obviously invalid to 0
        out[invalid] = 0.0
    
    return out


def _to_rgb_grid(x: torch.Tensor, nrow: int, take_first3: bool = True):
    """
    x: [B,C,H,W]. If C>3 and take_first3, visualize first 3 channels.
    Returns HxWx3 numpy image suitable for wandb.Image.
    NOTE: This is the ORIGINAL path used for W&B (kept unchanged),
          but now we also save PNGs with a unified grid helper below.
    """
    if x is None:
        return None
    vis = x
    if vis.ndim == 4 and vis.size(1) > 3 and take_first3:
        vis = vis[:, :3]
    grid = make_grid(vis.cpu(), nrow=nrow, normalize=True)  # (kept as-is)
    return grid.permute(1, 2, 0).numpy()


def _make_grid_tensor_for_png(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Build a grid tensor CHW in [0,1] with consistent parameters across val/test.
    - Enforces 3 channels by truncating if C>3 (same as test saver before).
    - Uses identical grid params so saved PNGs are comparable across phases.
    """
    if x is None:
        return None
    if x.ndim == 3:
        x = x.unsqueeze(0)
    k = min(k, x.size(0))
    vis = x[:k]
    if vis.size(1) > 3:  # enforce a 3-channel policy
        vis = vis[:, :3]
    grid = make_grid(
        vis.detach().cpu(),
        nrow=k,            # one row with k tiles
        padding=2,
        normalize=True,
        scale_each=True    # key to avoid grid-composition effects
    )
    return grid  # CHW, [0,1]


def _make_depth_grid_for_png(depth_tensor: torch.Tensor, k: int, depth_transformer, valid_mask=None) -> torch.Tensor:
    """
    Create a grayscale depth grid using magma-style visualization.
    Returns single-channel tensor ready for PNG saving.
    
    Args:
        depth_tensor: Depth tensor to visualize
        k: Number of samples to include
        depth_transformer: Transformer for denormalization
        valid_mask: Optional mask indicating valid pixels (True=valid, False=invalid)
    """
    if depth_tensor is None or depth_transformer is None:
        return None
    
    if depth_tensor.ndim == 3:
        depth_tensor = depth_tensor.unsqueeze(0)
    k = min(k, depth_tensor.size(0))
    depth_vis = depth_tensor[:k]
    
    # Handle valid_mask if provided
    mask_vis = None
    if valid_mask is not None:
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(0)
        mask_vis = valid_mask[:k]
    
    # Denormalize to meters and map for visualization  
    eval_min, eval_max = 2.0, 80.0
    depth_metric = depth_transformer.denormalize(depth_vis)
    depth_mapped = map_depth_for_vis(depth_metric, amin=eval_min, amax=eval_max, valid_mask=mask_vis)
    
    # Make grid (single channel)
    grid = make_grid(
        depth_mapped.detach().cpu(),
        nrow=k,
        padding=2,
        normalize=False,  # already in [0,1]
        scale_each=False
    )
    return grid  # CHW, [0,1]


def _save_png_grid(x: torch.Tensor, path: str, k: int):
    grid = _make_grid_tensor_for_png(x, k)
    if grid is None:
        return
    # grid is CHW already; save exactly what we show
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(grid, path)  # no extra normalize; grid is already [0,1]


class PoELogger(GenericLogger):
    """
    PoE-aware logger: logs inputs & reconstructions for BOTH modalities (event & depth).

    Validation:
      - keeps original W&B figures (unchanged)
      - ALSO saves PNGs using the SAME grid pipeline as test for 1:1 comparison

    Test:
      - saves PNGs for inputs, recons, priors, and cross-modal (same as before)
      - now uses unified grid builder under the hood for consistency
    """

    def __init__(
        self,
        train_log_img_freq: int = 1,
        train_log_score_freq: int = -1,
        train_ckpt_freq: int = -1,
        show_samples_at_start: bool = False,
        sampling_batch_size: int = 8,
        check_freq_via: str = "epoch",
        enable_save_ckpt: bool = True,
        save_test_dir: str = "/home/qd8/eval/poe",
        save_val_dir: str = None,  # new: where to write validation PNGs
    ):
        super().__init__(
            train_log_img_freq=train_log_img_freq,
            train_log_score_freq=train_log_score_freq,
            train_ckpt_freq=train_ckpt_freq,
            show_samples_at_start=show_samples_at_start,
            sampling_batch_size=sampling_batch_size,
            check_freq_via=check_freq_via,
            enable_save_ckpt=enable_save_ckpt,
        )
        self.save_test_dir = save_test_dir
        # default val dir next to test dir
        self.save_val_dir = save_val_dir or os.path.join(save_test_dir, "val_png")
        # depth transformer for proper depth visualization
        self.depth_transformer = None

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        super().on_fit_start(trainer, pl_module)
        self.depth_transformer = trainer.datamodule.depth_transform

    def visualize_batch(self, **batch):
        """
        Log a quick visual from a raw train batch at start.
        Tries common keys and handles tuple/list batches.
        """
        x_event, x_depth = None, None

        # dict-style batch
        if isinstance(batch, dict):
            x_event = batch.get("event") or batch.get("events") or batch.get("x_event") or batch.get("x")
            x_depth = batch.get("depth") or batch.get("image") or batch.get("rgb") or batch.get("y") or batch.get("target")
        # tuple/list-style batch: assume (inputs, target)
        elif isinstance(batch, (list, tuple)):
            if len(batch) >= 1 and isinstance(batch[0], torch.Tensor):
                x_event = batch[0]
            if len(batch) >= 2 and isinstance(batch[1], torch.Tensor):
                x_depth = batch[1]
        # bare tensor: treat as event input
        elif isinstance(batch, torch.Tensor):
            x_event = batch

        sizes = [t.size(0) for t in (x_event, x_depth) if isinstance(t, torch.Tensor)]
        if not sizes:
            return
        s = min(self.sampling_batch_size, min(sizes))

        logs = {}
        if isinstance(x_event, torch.Tensor):
            gi = _to_rgb_grid(x_event[:s], nrow=s, take_first3=True)  # keep original for W&B
            if gi is not None:
                logs["train/poe_sample_event_input"] = wandb.Image(gi)
        if isinstance(x_depth, torch.Tensor):
            gd = _to_rgb_grid(x_depth[:s], nrow=s, take_first3=False)  # depth currently 3ch RGB VAE
            if gd is not None:
                logs["train/poe_sample_depth_input"] = wandb.Image(gd)

        if logs:
            wandb.log(logs)

    # ----- optional abstract in base; keep no-op -----
    def log_score(self, pl_module, outputs):
        pass

    @hold_pbar("Visualizing PoE reconstructions…")
    @rank_zero_only
    def log_samples(self, trainer, pl_module, outputs):
        # Normalize 'outputs' to a list of dicts (GenericLogger passes a single dict per batch)
        if outputs is None:
            return
        if isinstance(outputs, dict):
            out_list = [outputs]
        elif isinstance(outputs, (list, tuple)):
            out_list = [o for o in outputs if isinstance(o, dict)]
        else:
            return

        if not out_list:
            return

        # Collect tensors across (possibly multiple) val batches
        ev_in, ev_rec, dp_in, dp_rec, etod, valid_masks = [], [], [], [], [], []
        for o in out_list:
            ei, er = o.get("event_input"), o.get("recon_event")
            di, dr = o.get("depth_input"), o.get("recon_depth")
            e2d = o.get("event2depth")
            vmask = o.get("valid_mask")
            if isinstance(ei, torch.Tensor): ev_in.append(ei)
            if isinstance(er, torch.Tensor): ev_rec.append(er)
            if isinstance(di, torch.Tensor): dp_in.append(di)
            if isinstance(dr, torch.Tensor): dp_rec.append(dr)
            if isinstance(e2d, torch.Tensor): etod.append(e2d)
            if isinstance(vmask, torch.Tensor): valid_masks.append(vmask)

        # Concatenate if present
        ev_in = torch.cat(ev_in, dim=0) if ev_in else None
        ev_rec = torch.cat(ev_rec, dim=0) if ev_rec else None
        dp_in = torch.cat(dp_in, dim=0) if dp_in else None
        dp_rec = torch.cat(dp_rec, dim=0) if dp_rec else None
        etod = torch.cat(etod, dim=0) if etod else None
        valid_mask = torch.cat(valid_masks, dim=0) if valid_masks else None

        s = self.sampling_batch_size

        # ---------------------------
        # (A) ORIGINAL W&B FIGURES – unchanged
        # ---------------------------
        # --- Event modality ---
        if ev_in is not None and ev_rec is not None:
            k = min(s, ev_in.size(0), ev_rec.size(0))
            gi = _to_rgb_grid(ev_in[:k], nrow=k, take_first3=True)
            gr = _to_rgb_grid(ev_rec[:k], nrow=k, take_first3=True)
            if gi is not None and gr is not None:
                fig, axs = plt.subplots(2, 1, figsize=(k * 3, 6))
                axs[0].imshow(gi); axs[0].set_title("Event Input (ch 0–2)"); axs[0].axis("off")
                axs[1].imshow(gr); axs[1].set_title("Event Recon (ch 0–2)"); axs[1].axis("off")
                plt.tight_layout()
                wandb.log({"val/event_reconstruction": wandb.Image(fig)})
                plt.close(fig)

        # --- Depth modality --- (using magma colormap like event logger)
        if dp_in is not None and dp_rec is not None and self.depth_transformer is not None:
            k = min(s, dp_in.size(0), dp_rec.size(0))
            
            # Handle valid_mask for visualization
            mask_k = None
            if valid_mask is not None:
                mask_k = valid_mask[:k]
            
            # Denormalize depth to meters
            dp_in_metric = self.depth_transformer.denormalize(dp_in[:k])
            dp_rec_metric = self.depth_transformer.denormalize(dp_rec[:k])
            
            # Map to [0,1] for visualization with mask (same range as event logger)
            eval_min, eval_max = 2.0, 80.0
            dp_in_vis = map_depth_for_vis(dp_in_metric, amin=eval_min, amax=eval_max, valid_mask=mask_k)
            dp_rec_vis = map_depth_for_vis(dp_rec_metric, amin=eval_min, amax=eval_max, valid_mask=mask_k)
            
            # Create grids for display
            dp_in_grid = make_grid(dp_in_vis, nrow=k, normalize=False)[0].detach().cpu().numpy()  # [H, W]
            dp_rec_grid = make_grid(dp_rec_vis, nrow=k, normalize=False)[0].detach().cpu().numpy()
            
            # Colorbar ticks (same as event logger)
            meter_ticks = torch.tensor([2.0, 5.0, 10.0, 20.0, 40.0, 80.0])
            tick_locs = map_depth_for_vis(meter_ticks, amin=eval_min, amax=eval_max).numpy()
            
            # Create figure with colorbars (2 rows × 2 cols: image + colorbar)
            fig = plt.figure(figsize=(k * 7, 12))
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.01], wspace=0.01, hspace=0.05)
            
            # Input row
            ax0 = fig.add_subplot(gs[0, 0])
            im0 = ax0.imshow(dp_in_grid, cmap='magma', vmin=0, vmax=1, aspect='auto')
            ax0.set_title('Depth Input')
            ax0.axis('off')
            cax0 = fig.add_subplot(gs[0, 1])
            cbar0 = fig.colorbar(im0, cax=cax0)
            cbar0.set_label('Depth (m)')
            cbar0.set_ticks(tick_locs)
            cbar0.set_ticklabels([f"{int(m)}" for m in meter_ticks])
            
            # Reconstruction row
            ax1 = fig.add_subplot(gs[1, 0])
            im1 = ax1.imshow(dp_rec_grid, cmap='magma', vmin=0, vmax=1, aspect='auto')
            ax1.set_title('Depth Recon')
            ax1.axis('off')
            cax1 = fig.add_subplot(gs[1, 1])
            cbar1 = fig.colorbar(im1, cax=cax1)
            cbar1.set_label('Depth (m)')
            cbar1.set_ticks(tick_locs)
            cbar1.set_ticklabels([f"{int(m)}" for m in meter_ticks])
            
            plt.tight_layout()
            wandb.log({"val/depth_reconstruction": wandb.Image(fig)})
            plt.close(fig)

        # ---------------------------
        # (B) NEW: SAVE PNGs FOR VAL with SAME PIPELINE AS TEST
        # ---------------------------
        os.makedirs(self.save_val_dir, exist_ok=True)
        # Event PNGs (unchanged)
        if ev_in is not None:
            _save_png_grid(ev_in,  os.path.join(self.save_val_dir, "val_event_input.png"), s)
        if ev_rec is not None:
            _save_png_grid(ev_rec, os.path.join(self.save_val_dir, "val_event_recon.png"), s)
        
        # Depth PNGs (using proper depth visualization)
        if dp_in is not None and self.depth_transformer is not None:
            depth_grid = _make_depth_grid_for_png(dp_in, s, self.depth_transformer, valid_mask)
            if depth_grid is not None:
                os.makedirs(os.path.dirname(os.path.join(self.save_val_dir, "val_depth_input.png")), exist_ok=True)
                save_image(depth_grid, os.path.join(self.save_val_dir, "val_depth_input.png"))
        if dp_rec is not None and self.depth_transformer is not None:
            depth_grid = _make_depth_grid_for_png(dp_rec, s, self.depth_transformer, valid_mask)
            if depth_grid is not None:
                save_image(depth_grid, os.path.join(self.save_val_dir, "val_depth_recon.png"))
        
        # Cross-modal (treat as depth if from event2depth)
        if etod is not None and self.depth_transformer is not None:
            depth_grid = _make_depth_grid_for_png(etod, s, self.depth_transformer, valid_mask)
            if depth_grid is not None:
                save_image(depth_grid, os.path.join(self.save_val_dir, "val_event2depth.png"))

        print(f"[PoELogger] epoch={trainer.current_epoch} | "
              f"event_present={ev_in is not None} depth_present={dp_in is not None} | "
              f"val_png_dir={self.save_val_dir}")

    @rank_zero_only
    def on_test_start(self, trainer, pl_module):
        """Ensure depth transformer is available for test visualization"""
        self.depth_transformer = trainer.datamodule.depth_transform

    @rank_zero_only
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int = 0):
        if outputs is None:
            return

        # normalize to list of dicts (mirror log_samples)
        if isinstance(outputs, dict):
            out_list = [outputs]
        elif isinstance(outputs, (list, tuple)):
            out_list = [o for o in outputs if isinstance(o, dict)]
        else:
            return
        if not out_list:
            return

        # collect across (possibly) multiple micro-batches
        ev_in, ev_rec, dp_in, dp_rec, valid_masks = [], [], [], [], []
        for o in out_list:
            if isinstance(o.get("event_input"), torch.Tensor):  ev_in.append(o["event_input"])
            if isinstance(o.get("recon_event"), torch.Tensor):  ev_rec.append(o["recon_event"])
            if isinstance(o.get("depth_input"), torch.Tensor):  dp_in.append(o["depth_input"])
            if isinstance(o.get("recon_depth"), torch.Tensor):  dp_rec.append(o["recon_depth"])
            if isinstance(o.get("valid_mask"), torch.Tensor):   valid_masks.append(o["valid_mask"])

        ev_in = torch.cat(ev_in, 0) if ev_in else None
        ev_rec = torch.cat(ev_rec, 0) if ev_rec else None
        dp_in = torch.cat(dp_in, 0) if dp_in else None
        dp_rec = torch.cat(dp_rec, 0) if dp_rec else None
        valid_mask = torch.cat(valid_masks, 0) if valid_masks else None

        k = self.sampling_batch_size
        save_dir = self.save_test_dir
        os.makedirs(save_dir, exist_ok=True)

        def _grid(x: torch.Tensor, nrow: int):
            if x is None: return None
            if x.ndim == 3: x = x.unsqueeze(0)
            n = min(nrow, x.size(0))
            vis = x[:n]
            if vis.size(1) > 3:  # enforce 3-ch policy like val visuals
                vis = vis[:, :3]
            return make_grid(vis.cpu(), nrow=n, normalize=True, scale_each=True)

        def _grid_bins(x: torch.Tensor, max_batches: int, bins_per_sample: int | None = None):
            """
            Make a grid where each row corresponds to one sample and columns are its bins
            (grayscale panels). If bins_per_sample is None, use all channels.
            """
            if x is None:
                return None
            if x.ndim == 3:
                x = x.unsqueeze(0)
            B, C, H, W = x.shape
            n = min(max_batches, B)
            x = x[:n]

            c_use = C if bins_per_sample is None else min(C, bins_per_sample)
            x = x[:, :c_use]  # [n, c_use, H, W]
            x = x.reshape(n * c_use, 1, H, W)  # [n*c_use, 1, H, W] -> grayscale tiles

            # nrow=c_use -> each row is one sample, columns are its bins
            return make_grid(x.cpu(), nrow=c_use, normalize=True, scale_each=True)


        # save PNGs (event)
        g = _grid(ev_in, k);
        save_image(g, os.path.join(save_dir, f"b{batch_idx:03d}_event_input.png")) if g is not None else None
        g = _grid(ev_rec, k);
        save_image(g, os.path.join(save_dir, f"b{batch_idx:03d}_event_recon.png")) if g is not None else None

        # ---- event: NEW per-bin strips (3 bins shown separately) ----
        g = _grid_bins(ev_in, k, bins_per_sample=3)
        save_image(g, os.path.join(save_dir, f"b{batch_idx:03d}_event_input_bins.png")) if g is not None else None

        g = _grid_bins(ev_rec, k, bins_per_sample=3)
        save_image(g, os.path.join(save_dir, f"b{batch_idx:03d}_event_recon_bins.png")) if g is not None else None

        # save PNGs (depth) - using proper depth visualization
        if dp_in is not None and self.depth_transformer is not None:
            g = _make_depth_grid_for_png(dp_in, k, self.depth_transformer, valid_mask)
            if g is not None:
                save_image(g, os.path.join(save_dir, f"b{batch_idx:03d}_depth_input.png"))
        if dp_rec is not None and self.depth_transformer is not None:
            g = _make_depth_grid_for_png(dp_rec, k, self.depth_transformer, valid_mask)
            if g is not None:
                save_image(g, os.path.join(save_dir, f"b{batch_idx:03d}_depth_recon.png"))

        # optional: priors / cross-modal
        for key in ["prior_event", "prior_depth", "event2depth"]:
            t = outputs.get(key)
            if isinstance(t, torch.Tensor):
                # Handle depth-related outputs with proper visualization and masking
                if "depth" in key and self.depth_transformer is not None:
                    g = _make_depth_grid_for_png(t, k, self.depth_transformer, valid_mask)
                else:
                    # Event-related outputs use standard grid
                    g = _grid(t, k)
                if g is not None:
                    save_image(g, os.path.join(save_dir, f"b{batch_idx:03d}_{key}.png"))

        print(f"[PoELogger/Test] b={batch_idx:03d} | "
              f"event_present={ev_in is not None} depth_present={dp_in is not None} | dir={save_dir}")
