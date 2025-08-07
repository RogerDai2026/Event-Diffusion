import os
import numpy as np
import torch
from tqdm import tqdm

def iter_latent_files(latent_dir, exts=(".npy", ".pt", ".pth")):
    for root, _, files in os.walk(latent_dir):
        for fname in files:
            if not fname.lower().endswith(exts):
                continue
            yield os.path.join(root, fname)

def load_latent_array(path):
    if path.lower().endswith(".npy"):
        z = np.load(path)
    else:
        z = torch.load(path, map_location="cpu")
        if isinstance(z, dict):
            for k in ("latent", "z", "code"):
                if k in z:
                    z = z[k]
                    break
        if hasattr(z, "cpu"):
            z = z.cpu().numpy()
        else:
            z = np.array(z)
    return z  # could be (1,3,H,W), (3,H,W), (H,W,3), etc.

def canonicalize_latent(z):
    """
    Return array in shape (..., H, W, C) where C=3.
    Handles:
      - (1, 3, H, W)
      - (3, H, W)
      - (H, W, 3)
    """
    if z.ndim == 4:
        # assume (B, C, H, W) or (1, C, H, W), merge batch
        z = z.reshape(-1, *z.shape[2:])  # (B*?, H, W) if C=1? but expect C=3 below
        # if it was (1,3,H,W), now (3,H,W) so fall through
    if z.ndim == 3:
        if z.shape[0] == 3:
            # (3, H, W) -> (H, W, 3)
            z = np.transpose(z, (1, 2, 0))
        elif z.shape[2] == 3:
            # already (H, W, 3)
            pass
        else:
            raise ValueError(f"Cannot canonicalize latent with shape {z.shape}")
    else:
        raise ValueError(f"Cannot canonicalize latent with shape {z.shape}")
    return z  # (H, W, 3)

def compute_per_channel_stats(latent_dir, max_files=None):
    sum_channels = None  # shape (3,)
    sumsq_channels = None
    total_pixels = 0

    paths = list(iter_latent_files(latent_dir))
    if max_files:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(f"No latent files found under {latent_dir}")

    for path in tqdm(paths, desc="Accumulating per-channel stats", unit="file"):
        try:
            z = load_latent_array(path)
            z = canonicalize_latent(z)  # now (H,W,3)
        except Exception as e:
            print(f"skip {path}: {e}")
            continue

        H, W, C = z.shape
        if C != 3:
            raise ValueError(f"Unexpected channel size {C} in {path}, expected 3")

        pixels = z.reshape(-1, C)  # (H*W, 3)
        if sum_channels is None:
            sum_channels = np.zeros(C, dtype=np.float64)
            sumsq_channels = np.zeros(C, dtype=np.float64)

        sum_channels += pixels.sum(axis=0)
        sumsq_channels += (pixels ** 2).sum(axis=0)
        total_pixels += pixels.shape[0]

    if total_pixels == 0:
        raise RuntimeError("No valid pixels processed.")

    mean = sum_channels / total_pixels  # (3,)
    second_moment = sumsq_channels / total_pixels
    var = second_moment - mean ** 2
    var = np.clip(var, a_min=1e-12, a_max=None)
    scale_per_channel = 1.0 / np.sqrt(var)

    print("Per-channel statistics:")
    for i in range(3):
        print(f"  Channel {i}: mean={mean[i]:.6f}, var={var[i]:.6f}, scale={scale_per_channel[i]:.6f}")

    return mean, scale_per_channel

def normalize_latent(latent, mean, scale):
    z = canonicalize_latent(latent)  # (H,W,3)
    return (z - mean.reshape((1,1,-1))) * scale.reshape((1,1,-1))


if __name__ == "__main__":
    latent_dir = "/shared/qd8/vae_latent_nbin3/train/latents"
    mean, scale = compute_per_channel_stats(latent_dir, max_files=None)
    # example application
    example_path = next(iter(iter_latent_files(latent_dir)))
    latent_example = load_latent_array(example_path)
    normalized = normalize_latent(latent_example, mean, scale)
    # Save params for reuse
    # np.save("event_latent_mean.npy", mean)
    # np.save("event_latent_scale.npy", scale)
