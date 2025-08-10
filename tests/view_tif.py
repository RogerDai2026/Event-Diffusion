#!/usr/bin/env python3
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

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
            scaled = (channel - vmin) / (vmax - vmin)  # [0,1]
            scaled = (scaled * 255.0).round().clip(0, 255).astype(np.uint8)
            out[b] = scaled
    return out


def main():
    path = '/shared/qd8/event3d/MVSEC/outdoor_day2_vox5/0000000039.npz'
    dst_dir = "/shared/qd8/event3d/MVSEC/"
    os.makedirs(dst_dir, exist_ok=True)
    dst_tif = os.path.join(dst_dir, "event_tensor_0000003562.tif")

    # stack into shape [N, nbin, H, W]:
    # img = tifffile.imread(path)
    # img = np.load(path)

    with np.load(path, allow_pickle=False) as f:
        print(f.files)  # -> ['vox']
        img = f["vox"]  # numpy array
    print("Loaded:", path)
    print("Shape:", img.shape, "dtype:", img.dtype)
    #
    # convert to uint8 for visualization
    vis = float_to_uint8_nbin(img)  # shape (nbin, H, W), uint8

    # tifffile.imwrite(dst_tif, vis, bigtiff=True)
    # print(f"Saved converted NBIN image to {dst_tif}")

    # Per-bin stats (on original floats and after scaling)
    for b in range(img.shape[0]):
        channel = img[b]
        nz = np.count_nonzero(channel)
        total = channel.size
        unique = np.unique(channel)
        print(f"[float] Bin {b}: min={channel.min():.4f}, max={channel.max():.4f}, sum={float(channel.sum()):.4f}, nonzero={nz}/{total}, unique_sample={unique[:10]}{'...' if unique.size>10 else ''}")
        if nz > 0:
            coords = np.argwhere(channel != 0)
            sample_vals = [float(channel[tuple(c)]) for c in coords[:5]]
            print(f"    sample nonzero coords: {coords[:5].tolist()} values: {sample_vals}")

        uchannel = vis[b]
        unique_u = np.unique(uchannel)
        print(f"[uint8] Bin {b}: min={int(uchannel.min())}, max={int(uchannel.max())}, unique_sample={unique_u[:10]}{'...' if unique_u.size>10 else ''}")

    if np.count_nonzero(img) == 0:
        print("WARNING: ENTIRE IMAGE IS ZERO")

    # Visualization
    fig, axs = plt.subplots(vis.shape[0], 1, figsize=(4, 2 * vis.shape[0]))
    for b in range(vis.shape[0]):
        ax = axs[b] if vis.shape[0] > 1 else axs
        im = ax.imshow(vis[b], cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax.set_title(f"Bin {b}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.01)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
