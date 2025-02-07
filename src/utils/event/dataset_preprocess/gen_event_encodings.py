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



# TIME_ENCODING = None

def nbin_encoding(times: ndarray, polarity: ndarray, x: ndarray, y: ndarray, args: dict, nbin: int) -> ndarray:
    x = x.astype(np.int64)
    y = y.astype(np.int64)
    normalized_time = normalize(times)
    encoded_image = np.zeros((nbin, args.height, args.width), dtype=np.int8)
    time_bin = np.minimum((normalized_time * nbin).astype(int), nbin - 1)
    for b in range(nbin):
        cur_frame = np.zeros((args.height, args.width), dtype=np.int8)
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


def save_from_discrete_event_frames(base_dir, event_dir, save_dir, time_encoding, args):

    # handle output directory
    save_dir = os.path.join(save_dir, time_encoding)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # else:
    #     if os.listdir(save_dir):
    #         print("Output directory is not empty. Exiting.")
    #         return

    # read input list
    npy_files = sorted(glob.glob(os.path.join(base_dir, event_dir, "*.npy")))
    if len(npy_files) == 0:
        print("No npy files found. Trying to load npz files")
        npy_files = sorted(glob.glob(os.path.join(base_dir, event_dir, "*.npz")))
        if len(npy_files) == 0: # why is this needed?
            raise FileNotFoundError(f"No npy/npz files found in the directory {os.path.join(base_dir, event_dir)}")
    print(f"Found {len(npy_files)} npy files")

    npy_counter = 0
    for npy_file in tqdm(npy_files, desc="Processing npy/npz files", total=len(npy_files), dynamic_ncols=True,
                         ascii=True):
        try:
            data = np.load(npy_file)
            times = data[:,0].squeeze()/1e6
            x = data[:,1].squeeze()
            y = data[:,2].squeeze()
            polarity = data[:,3].squeeze()
            #x, y, times, and polarity are all flat arrays
        except Exception as e: # TODO exception is too broad
            data = np.load(npy_file)
            times = data["t"].squeeze() / 1e6
            x = data["x"].squeeze()
            y = data["y"].squeeze()
            polarity = data["p"].squeeze()

        if time_encoding == 'LINEAR':
            raise NotImplementedError()
        elif time_encoding == 'LOG':
            raise NotImplementedError()
        elif time_encoding == 'PYRAMIDAL':
            raise NotImplementedError()
        elif time_encoding == 'POSITIONAL':
            raise NotImplementedError()
        elif time_encoding == 'VAE_ROBUST':
            raise NotImplementedError()
        elif time_encoding == 'N_BINS_5':
            encoded_img = nbin_encoding(times, polarity, x, y, args, nbin=5)
        else:
            raise ValueError(f"Invalid time encoding: {time_encoding}")

        # save image
        encoded_img = np.int8(encoded_img)
        savename = os.path.join(save_dir, f"event_frame_{npy_counter:04d}.tif")
        npy_counter += 1
        io.imwrite(savename, encoded_img)

    # print-summary
    print(f"Processed {npy_counter} npy files")
    print(f"Saved event frames to {save_dir}")
    return

@hydra.main(version_base="1.3", config_path="../../../../configs/", config_name="gen_event_encoding.yaml")
def main(cfg: DictConfig):
    # global TIME_ENCODING
    # TIME_ENCODING = cfg.data.TIME_ENCODING
    save_event_frames(cfg.base_dir, cfg.event_dir, cfg.save_dir, cfg.time_encoding, cfg)

if __name__ == "__main__":
    main()