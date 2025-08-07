import os
import torch
from torch.utils.data import Dataset
from src.utils.event.depth_transform import DepthNormalizerBase
from typing import Union

class CachedLatentDepthDataset(Dataset):
    """
    PyTorch Dataset to load pre-saved VAE latents and corresponding depth tensors.
    Expects directory structure:
      root_dir/
        train/
          latents/   # *.pt files each containing a tensor [1, C, H, W] or [C, H, W]
          depths/    # *.pt files each containing a tensor [1, 1, H, W] or [1, H, W]
        val/
          latents/
          depths/
    """

    def __init__(self, root_dir: str, split: str = "train", depth_transform: Union[DepthNormalizerBase, None] = None,):
        self.latent_dir = os.path.join(root_dir, split, "latents")
        self.depth_dir  = os.path.join(root_dir, split, "depths")
        self.depth_transform: DepthNormalizerBase = depth_transform
        # gather all latent filenames ending with '_latent.pt'
        latent_files = sorted(
            f for f in os.listdir(self.latent_dir) if f.endswith("_latent.pt")
        )
        # derive ids by stripping the suffix
        self.ids = [fname.replace("_latent.pt", "") for fname in latent_files]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        base = self.ids[idx]
        # load latent tensor
        latent_path = os.path.join(self.latent_dir, base + "_latent.pt")
        z = torch.load(latent_path)
        # remove leading batch dim if present
        if z.ndim == 4 and z.shape[0] == 1:
            z = z.squeeze(0)

        # load depth tensor
        depth_path = os.path.join(self.depth_dir, base + "_depth.pt")
        d = torch.load(depth_path)
        # remove leading batch dim if present
        if d.ndim == 4 and d.shape[0] == 1:
            d = d.squeeze(0)
        # ensure depth has shape [1, H, W]
        if d.ndim == 2:
            d = d.unsqueeze(0)

        return {
            "rgb_norm":      z.float(),       # latent tensor
            "depth_raw_norm": d.float()       # depth tensor
        }
