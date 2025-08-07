#!/usr/bin/env python3
"""
Script to precompute and save VAE latents (and downsampled depth) for event data.
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from hydra import initialize, compose
from tqdm.auto import tqdm
import numpy as np
import tifffile
from omegaconf import DictConfig

from src.data.event_datamodule import EventDataModule
from src.models.baselines.cnn.vae_module import LatentVAELitModule

# === User settings: update these paths before running ===
VAE_CHECKPOINT = "/home/qd8/models/Event-WassDiff/2025-07-26_17-07-57/VAE/bel8esnf/checkpoints/epoch=29-step=51030.ckpt"
VAE_CFG_PATH  = "src/utils/latent_diffusion/models/first_stage_models/kl-f4/config.yaml"
OUTPUT_ROOT   = "/shared/qd8/vae_latent_nbin3" # where latents & depths will be saved

# Hydra data-config name (in configs/data directory)
HYDRA_DATA_CONFIG = "event_custom_nbin3"
# Splits to process
SPLITS = ["train", "val"]
CUDA_DEVICE_ID = 4

def main():
    # 1) Load data module via Hydra
    device = torch.device(f"cuda:{CUDA_DEVICE_ID}" if torch.cuda.is_available() else "cpu")

    with initialize(version_base=None, config_path="../../configs/data", job_name="vae_latents"):
        data_cfg = compose(config_name=HYDRA_DATA_CONFIG)

    dm = EventDataModule(
        data_config=data_cfg.data_config,
        augmentation_args=data_cfg.augmentation_args,
        depth_transform_args=data_cfg.depth_transform_args,
        batch_size=1,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        seed=data_cfg.seed,
    )
    dm.setup(stage="fit")

    # 2) Instantiate and load pretrained VAE
    vae: LatentVAELitModule = LatentVAELitModule.load_from_checkpoint(
        checkpoint_path=VAE_CHECKPOINT,
        map_location="cpu",
        strict=False,
        cfg_path=VAE_CFG_PATH,
        allow_resize=True,
        compile=False,
    )
    vae.eval().to(device)

    # 3) Iterate splits and cache
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for split in SPLITS:
        dataset = getattr(dm, f"{split}_dataset")
        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            shuffle=False,
        )
        total = len(loader)
        out_lat = os.path.join(OUTPUT_ROOT, split, "latents")
        out_dep = os.path.join(OUTPUT_ROOT, split, "depths")
        os.makedirs(out_lat, exist_ok=True)
        os.makedirs(out_dep, exist_ok=True)

        print(f"Caching split '{split}' → {len(loader)} samples")

        expected_hw = (180,320)
        skipped = 0

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(loader, desc=f"{split} latents", total=total)):
                # encode events
                events = batch["rgb_norm"].to(device)
                # print(f"event_shape{events.shape}")
                post = vae.autoencoder.encode(events)
                z = post.sample().cpu()  # [1, embed_dim, H', W']
                _, _, H, W = z.shape

                if (H, W) == expected_hw:
                    torch.save(z, os.path.join(out_lat, f"{idx:06d}_latent.pt"))
                else:
                    skipped += 1
                    print(f"  [skip] {split} sample {idx:06d}: got ({H}×{W}), expected {expected_hw}")
                    continue

                # print latent size
                # print(f"Sample {idx:06d}: latent shape {z.shape}")
                # save latent tensor as .pt

                # downsample GT depth and save
                depth = batch["depth_raw_norm"]  # [1,1,H,W]
                depth_small = F.interpolate(
                    depth,
                    size=z.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).cpu()
                # print(f"Sample {idx:06d}: depth shape {depth_small.shape}")
                torch.save(depth_small, os.path.join(out_dep, f"{idx:06d}_depth.pt"))

    print("✅ Caching complete.")


if __name__ == "__main__":
    main()
