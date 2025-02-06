from typing import Any, Dict, List, Optional, Tuple
import os
import glob
import hydra
import rootutils
from tqdm import tqdm
import numpy as np
import imageio.v3 as io
from omegaconf import DictConfig
from src.utils.event.dataset_preprocess.vkitti.generate_event_dataset import (
    get_pyramidal_encoding, get_vae_robust_encoding, positional_encoding_1d, normalize)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

TIME_ENCODING = None

def get_new_encoding(times, polarity, x, y, image):
    pass

def save_event_frames(base_dir, event_dir, save_dir, time_encoding, args):
    height = args.get("height", None)
    width = args.get("width", None)
    if height is None or width is None:
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
        height = img.shape[0]
        width = img.shape[1]
        print(f"Height: {height}, Width: {width}")
    if args.get("npy"):
        save_from_discrete_event_frames(base_dir, event_dir, save_dir, time_encoding, height, width, args)
    else:
        raise NotImplementedError()
        # save_from_continous_events(base_dir, event_dir, save_dir, time_encoding, height, width, args)


def save_from_discrete_event_frames(base_dir, event_dir, save_dir, time_encoding, height, width, args):
    # Get all the npy files in the directory
    npy_files = sorted(glob.glob(os.path.join(base_dir, event_dir, "*.npy")))
    if len(npy_files) == 0:
        print("No npy files found. Trying to load npz files")
        npy_files = sorted(glob.glob(os.path.join(base_dir, event_dir, "*.npz")))
        if len(npy_files) == 0:
            raise FileNotFoundError(f"No npy/npz files found in the directory {os.path.join(base_dir, event_dir)}")

    images = []
    for npy_file in tqdm(npy_files, desc="Processing npy/npz files", total=len(npy_files),
                         dynamic_ncols=True, ascii=True):
        # Load the npz file
        try:
            data = np.load(npy_file)
            times = data[:, 0].squeeze() / 1e6
            x = data[:, 1].squeeze()
            y = data[:, 2].squeeze()
            polarity = data[:, 3].squeeze()
        except:
            data = np.load(npy_file)
            times = data["t"].squeeze() / 1e6
            x = data["x"].squeeze()
            y = data["y"].squeeze()
            polarity = data["p"].squeeze()

        vmin = args.get("vmin", None)
        vmax = args.get("vmax", None)
        interval_vmax = args.get("interval_vmax", None)

        # First encode time as color
        if TIME_ENCODING["LINEAR"] == time_encoding:
            # Normalize the time to be between 0 and 1
            times = normalize(times, relative_vmin=vmin, relative_vmax=vmax, interval_vmax=interval_vmax)
        elif TIME_ENCODING["LOG"] == time_encoding:
            # TODO: Not implemented properly
            # Normalize the time to be between 0 and 1
            times = normalize(np.log(times + 1), interval_vmax=vmax)
        elif TIME_ENCODING["POSITIONAL"] == time_encoding:
            times = positional_encoding_1d(times)
        elif (TIME_ENCODING["PYRAMIDAL"] == time_encoding):
            # Normalize the time to be between 0 and 1
            times = normalize(times, relative_vmin=vmin, relative_vmax=vmax, interval_vmax=interval_vmax)
            L0, L1, L2 = get_pyramidal_encoding(times, polarity, x, y)
        elif (TIME_ENCODING["VAE_ROBUST"] == time_encoding):
            # Normalize the time to be between 0 and 1
            image = np.zeros((height, width, 3), dtype=np.float32) + 0.5
            times = normalize(times, relative_vmin=vmin, relative_vmax=vmax, interval_vmax=interval_vmax)
            event_image = get_vae_robust_encoding(times, polarity, x, y, image)

        # Initialize the image as all ones
        if (TIME_ENCODING["PYRAMIDAL"] == time_encoding):
            image = np.zeros((height, width, 3))
            image = image + 0.5

            neg_p = polarity.min()

            xx = L0["x"]
            yy = L0["y"]
            pol = L0["polarity"]
            t = L0["times"]
            image[np.int64(yy[pol == neg_p]), np.int64(xx[pol == neg_p]), 2] = (np.zeros_like(t))[pol == neg_p]
            image[np.int64(yy[pol == 1]), np.int64(xx[pol == 1]), 2] = (np.ones_like(t))[pol == 1]

            xx = L1["x"]
            yy = L1["y"]
            pol = L1["polarity"]
            t = L1["times"]
            image[np.int64(yy[pol == neg_p]), np.int64(xx[pol == neg_p]), 1] = (np.zeros_like(t))[pol == neg_p]
            image[np.int64(yy[pol == 1]), np.int64(xx[pol == 1]), 1] = (np.ones_like(t))[pol == 1]

            xx = L2["x"]
            yy = L2["y"]
            pol = L2["polarity"]
            t = L2["times"]
            image[np.int64(yy[pol == neg_p]), np.int64(xx[pol == neg_p]), 0] = (np.zeros_like(t))[pol == neg_p]
            image[np.int64(yy[pol == 1]), np.int64(xx[pol == 1]), 0] = (np.ones_like(t))[pol == 1]
        elif (TIME_ENCODING["VAE_ROBUST"] == time_encoding):
            image = event_image
        else: # only verified for linear
            image = np.zeros((height, width, 3))
            # Assign the red channel to the 0 polarity events, and make its blue and green channels zero
            times = times
            neg_p = polarity.min() # either 0 or -1
            image[np.int64(y[polarity == neg_p]), np.int64(x[polarity == neg_p]), 0] = times[polarity == neg_p]
            # Assign the blue channel to the 1 polarity events
            image[np.int64(y[polarity == 1]), np.int64(x[polarity == 1]), 2] = times[polarity == 1]

        # ones = np.ones_like(polarity)
        # image = np.ones((height, width, 3))
        # image[np.int64(y[polarity == -1]), np.int64(x[polarity == -1]), 0] = ones[polarity == -1]
        # image[np.int64(y[polarity == 1]), np.int64(x[polarity == 1]), 2] = ones[polarity == 1]

        images.append(image)

    # Save the images to disk
    total_images = len(images)
    print(f"Saving {total_images} images to disk")

    # Check first if the directory exists
    save_dir = os.path.join(save_dir, args.get("time_encoding", "LINEAR"))
    print(f"Saving images to {save_dir}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for iter in tqdm(range(total_images), desc="Saving images",
                     dynamic_ncols=True, ascii=True):
        img = np.uint8(255 * images[iter])
        savename = os.path.join(save_dir, f"event_frame_{iter:04d}.tif")
        io.imwrite(savename, img)

    # Also write the configuration file
    # config = configparser.ConfigParser()
    # config["EventDataset"] = {
    #     "height": str(height),
    #     "width": str(width),
    #     "time_encoding": args.get("time_encoding", "LINEAR"),
    #     "vmin": str(vmin),
    #     "vmax": str(vmax),
    #     "interval_vmax": str(interval_vmax),
    # }
    # # Write the configuration file
    # with open(os.path.join(save_dir, "config.ini"), 'w') as configfile:
    #     config.write(configfile)

@hydra.main(version_base="1.3", config_path="../../../../configs/", config_name="gen_event_encoding.yaml")
def main(cfg: DictConfig):
    global TIME_ENCODING
    TIME_ENCODING = cfg.data.TIME_ENCODING
    save_event_frames(cfg.base_dir, cfg.event_dir, cfg.save_dir, cfg.time_encoding, cfg)

if __name__ == "__main__":
    main()