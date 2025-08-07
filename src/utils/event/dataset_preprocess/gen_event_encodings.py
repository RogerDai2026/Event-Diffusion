from typing import Any, Dict, List, Optional, Tuple
import os
import glob
import hydra
import rootutils
from numpy import ndarray
from tqdm import tqdm
import numpy as np
import imageio.v3 as io
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.event.dataset_preprocess.vkitti.generate_event_dataset import (
    get_pyramidal_encoding, get_vae_robust_encoding, positional_encoding_1d, normalize)


def normalize(x, relative_vmin=None, relative_vmax=None, interval_vmax=None):
    vmax = x.max()
    vmin = x.min()
    if (relative_vmax is not None):
        vmax = relative_vmax + vmin
    if (relative_vmin is not None):
        vmin = relative_vmin + vmin
    if (interval_vmax is None):
        interval_vmax = vmax - vmin


    # Keep only the values between vmin and vmax
    x = x * (x >= vmin) * (x <= vmax)

    return (x - vmin) / interval_vmax

# TIME_ENCODING = None

def nbin_encoding(times: ndarray, polarity: ndarray, x: ndarray, y: ndarray, args: dict, nbin: int) -> ndarray:
    x = x.astype(np.int64)
    y = y.astype(np.int64)
    normalized_time = normalize(times)
    polarity[polarity == 1] = 255
    polarity[polarity == -1] = 0
    encoded_image = np.ones((nbin, args.height, args.width), dtype=np.int8) * 128
    time_bin = np.minimum((normalized_time * nbin).astype(int), nbin - 1)
    for b in range(nbin):
        cur_frame = np.ones((args.height, args.width), dtype=np.int8) * 128
        mask = (time_bin == b)
        cur_frame[y[mask], x[mask]] = polarity[mask]
        encoded_image[b] = cur_frame
    return encoded_image


def latest_arrival_encoding(times, polarity, x, y, n_latest_snapshots=3):
    pass

def count_n_latest_arrival_encoding(times, polarity, x, y):
    pass

def save_event_frames(base_dir, event_dir, save_dir, time_encoding, args):
    if args.height is None or args.width is None:
        # get the height and width from the first image
        if args.get("npy"):
            print("Reading from npy files")
            image_files = glob.glob(os.path.join(base_dir, "../depth/frames/", "*.png"))
        else:
            print("Reading from txt events")
            image_files = glob.glob(os.path.join(base_dir, "depth/Camera_0/", "*.png"))
        if len(image_files) == 0:
            image_files = glob.glob(os.path.join(base_dir, event_dir, "*.tif"))
            if len(image_files) == 0:
                raise FileNotFoundError(f"No images found in the directory {os.path.join(base_dir, event_dir)}")
        # Read the first image to get the height and width
        img = io.imread(image_files[0])
        args.height = img.shape[0]
        args.width = img.shape[1]
        print(f"Height: {args.height}, Width: {args.width}")
    if args.get("npy"):
        save_from_discrete_event_frames(base_dir, event_dir, save_dir, time_encoding, args)
    else:
        raise NotImplementedError()
        # save_from_continous_events(base_dir, event_dir, save_dir, time_encoding, height, width, args)

#
# def save_from_discrete_event_frames(base_dir, event_dir, save_dir, time_encoding, args):
#
#     # handle output directory
#     save_dir = os.path.join(save_dir, time_encoding)
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     # else:
#     #     if os.listdir(save_dir):
#     #         print("Output directory is not empty. Exiting.")
#     #         return
#
#     # read input list
#     npy_files = sorted(glob.glob(os.path.join(base_dir, event_dir, "*.npy")))
#     if len(npy_files) == 0:
#         print("No npy files found. Trying to load npz files")
#         npy_files = sorted(glob.glob(os.path.join(base_dir, event_dir, "*.npz")))
#         if len(npy_files) == 0: # why is this needed?
#             raise FileNotFoundError(f"No npy/npz files found in the directory {os.path.join(base_dir, event_dir)}")
#     print(f"Found {len(npy_files)} npy files")
#
#     npy_counter = 0
#     for npy_file in tqdm(npy_files, desc="Processing npy/npz files", total=len(npy_files), dynamic_ncols=True,
#                          ascii=True):
#         try:
#             data = np.load(npy_file)
#             times = data[:,0].squeeze()/1e6
#             x = data[:,1].squeeze()
#             y = data[:,2].squeeze()
#             polarity = data[:,3].squeeze()
#             #x, y, times, and polarity are all flat arrays
#         except Exception as e: # TODO exception is too broad
#             data = np.load(npy_file)
#             times = data["t"].squeeze() / 1e6
#             x = data["x"].squeeze()
#             y = data["y"].squeeze()
#             polarity = data["p"].squeeze()
#
#         if time_encoding == 'LINEAR':
#             raise NotImplementedError()
#         elif time_encoding == 'LOG':
#             raise NotImplementedError()
#         elif time_encoding == 'PYRAMIDAL':
#             raise NotImplementedError()
#         elif time_encoding == 'POSITIONAL':
#             raise NotImplementedError()
#         elif time_encoding == 'VAE_ROBUST':
#             raise NotImplementedError()
#         elif time_encoding == 'N_BINS_5':
#             encoded_img = nbin_encoding(times, polarity, x, y, args, nbin=5)
#         else:
#             raise ValueError(f"Invalid time encoding: {time_encoding}")
#
#         # save image
#         encoded_img = np.uint8(encoded_img)
#         savename = os.path.join(save_dir, f"event_frame_{npy_counter:04d}.tif")
#         npy_counter += 1
#         io.imwrite(savename, encoded_img)
#
#     # print-summary
#     print(f"Processed {npy_counter} npy files")
#     print(f"Saved event frames to {save_dir}")
#     return



def save_from_discrete_event_frames(base_dir: str,
                                    event_folder_name: str,
                                    save_dir: str,
                                    time_encoding: str,
                                    args: DictConfig):
    """
    Walk scene_* dirs under base_dir/event_folder_name, skip tiny or corrupted .npz,
    encode each remaining file, and dump to save_dir/time_encoding/.
    """
    out_dir = os.path.join(save_dir, time_encoding)
    os.makedirs(out_dir, exist_ok=True)

    scene_dirs = sorted(glob.glob(os.path.join(base_dir, "scene_*")))
    # count total for progress bar
    total = sum(len(glob.glob(os.path.join(sd, event_folder_name, "*.npz")))
                for sd in scene_dirs)
    pbar = tqdm(total=total, desc="Encoding events", dynamic_ncols=True)
    idx = 0

    for scene_dir in scene_dirs:
        ev_dir = os.path.join(scene_dir, event_folder_name)
        npz_paths = sorted(glob.glob(os.path.join(ev_dir, "*.npz")))

        for npz_path in npz_paths:

            # 2) Try to load .npz and access keys
            try:
                data = np.load(npz_path)
                keys = list(data.keys())  # this may still throw on corruption
                # pick your events array
                if "events" in data:
                    ev = data["events"]
                else:
                    ev = data[keys[0]]
            except Exception as e:
                print(f"⚠️  Skipping corrupted file {npz_path}: {e}")
                pbar.update(1)
                continue

            # 3) Unpack fields (structured or fallback)
            try:
                if ev.dtype.names is not None:
                    t = ev["t"].astype(np.float64) / 1e6
                    x = ev["x"].astype(np.int64)
                    y = ev["y"].astype(np.int64)
                    p = ev["pol"].astype(np.int8)
                else:
                    arr = ev
                    t = arr[:, 0] / 1e6
                    x = arr[:, 1].astype(np.int64)
                    y = arr[:, 2].astype(np.int64)
                    p = arr[:, 3].astype(np.int8)
            except Exception as e:
                print(f"⚠️  Skipping malformed events in {npz_path}: {e}")
                pbar.update(1)
                continue

            # 4) Encode into n‐bin representation
            try:
                encoded = nbin_encoding(t, p, x, y, args, nbin=5)
            except Exception as e:
                print(f"⚠️  Error encoding {npz_path}: {e}")
                pbar.update(1)
                continue

            # 5) Write out TIFF
            out_path = os.path.join(out_dir, f"event_frame_{idx:06d}.tif")
            try:
                tifffile.imwrite(out_path, encoded, bigtiff=True)
            except OSError as e:
                print(f"⚠️  Skipping write {out_path}: {e}")
                pbar.update(1)
            continue

            idx += 1
            pbar.update(1)

    pbar.close()
    print(f"\nDone: wrote {idx} encoded frames to {out_dir}")


@hydra.main(version_base="1.3", config_path="../../../../configs/", config_name="gen_event_encoding_custom.yaml")
def main(cfg: DictConfig):
    # global TIME_ENCODING
    # TIME_ENCODING = cfg.data.TIME_ENCODING
    save_event_frames(cfg.base_dir, cfg.event_dir, cfg.save_dir, cfg.time_encoding, cfg)

if __name__ == "__main__":
    main()