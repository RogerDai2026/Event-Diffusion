# Last modified: 2024-04-30
#
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# If you use or adapt this code, please attribute to https://github.com/prs-eth/marigold.
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

from typing import Optional
import io
import os
import random
import tarfile
from enum import Enum
from typing import Union
import numpy as np
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize
import tifffile # added
from src.utils.event.depth_transform import DepthNormalizerBase


class DatasetMode(Enum):
    RGB_ONLY = "rgb_only"
    EVAL = "evaluate"
    TRAIN = "train"


class DepthFileNameMode(Enum):
    """Prediction file naming modes"""

    id = 1  # id.png
    rgb_id = 2  # rgb_id.png
    i_d_rgb = 3  # i_d_1_rgb.png
    rgb_i_d = 4


def read_image_from_tar(tar_obj, img_rel_path):
    image = tar_obj.extractfile("./" + img_rel_path)
    image = image.read()
    image = Image.open(io.BytesIO(image))


class BaseDepthDataset(Dataset):
    def __init__(
        self,
        mode: DatasetMode,
        filename_ls_path: str,
        dataset_dir: str,
        disp_name: str,
        min_depth: float,
        max_depth: float,
        has_filled_depth: bool,
        name_mode: DepthFileNameMode,
        depth_transform: Union[DepthNormalizerBase, None] = None,
        augmentation_args: dict = None,
        resize_to_hw=None,
        move_invalid_to_far_plane: bool = True,
        rgb_transform=lambda x: x / 255.0 * 2 - 1,  #  [0, 255] -> [-1, 1],
        io_args: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        # dataset info
        self.filename_ls_path = filename_ls_path
        self.dataset_dir = dataset_dir
        assert os.path.exists(
            self.dataset_dir
        ), f"Dataset does not exist at: {self.dataset_dir}"
        self.disp_name = disp_name
        self.has_filled_depth = has_filled_depth
        self.name_mode: DepthFileNameMode = name_mode
        self.min_depth = min_depth
        self.max_depth = max_depth

        # training arguments
        self.depth_transform: DepthNormalizerBase = depth_transform
        self.augm_args = augmentation_args
        self.resize_to_hw = resize_to_hw
        self.rgb_transform = rgb_transform
        self.move_invalid_to_far_plane = move_invalid_to_far_plane

        # additional arguments
        self.read_event_via = io_args['read_event_via']

        # Load filenames
        with open(self.filename_ls_path, "r") as f:
            self.filenames = [
                s.split() for s in f.readlines()
            ]  # [['rgb.png', 'depth.tif'], [], ...]

        # Tar dataset
        self.tar_obj = None
        self.is_tar = (
            True
            if os.path.isfile(dataset_dir) and tarfile.is_tarfile(dataset_dir)
            else False
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        try:
            rasters, other = self._get_data_item(index)
        except Exception as e:
            print(f"Error loading data at index {index}: {e}")
            raise e  # Re-raise the exception instead of continuing
        
        # if DatasetMode.TRAIN == self.mode: #or DatasetMode.EVAL == self.mode:
        rasters = self._training_preprocess(rasters)
        # merge
        outputs = rasters
        outputs.update(other)
        return outputs

    def _get_data_item(self, index):
        rgb_rel_path, depth_rel_path, filled_rel_path = self._get_data_path(index=index)

        rasters = {}

        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        # Depth data
        # if DatasetMode.RGB_ONLY != self.mode:
        # load data
        depth_data = self._load_depth_data(
            depth_rel_path=depth_rel_path, filled_rel_path=filled_rel_path
        )
        rasters.update(depth_data)
        # valid mask
        rasters["valid_mask_raw"] = self._get_valid_mask(
            rasters["depth_raw_linear"]
        ).clone()
        rasters["valid_mask_filled"] = self._get_valid_mask(
            rasters["depth_filled_linear"]
        ).clone()

        other = {"index": index, "rgb_relative_path": rgb_rel_path}

        return rasters, other

    def _load_rgb_data(self, rgb_rel_path):
        # Read RGB data
        rgb = self._read_rgb_file(rgb_rel_path)
        # Note: For mvsec and new nbin encoding, comment this out
        # rgb_norm = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb

        outputs = {
            "rgb_int": torch.from_numpy(rgb).int(),
            "rgb_norm": torch.from_numpy(rgb_norm).float(),
        }
        return outputs

    def _load_depth_data(self, depth_rel_path, filled_rel_path):
        # Read depth data
        outputs = {}
        depth_raw = self._read_depth_file(depth_rel_path).squeeze()
        depth_raw_linear = torch.from_numpy(depth_raw).float().unsqueeze(0)  # [1, H, W]
        outputs["depth_raw_linear"] = depth_raw_linear.clone()

        if self.has_filled_depth:
            depth_filled = self._read_depth_file(filled_rel_path).squeeze()
            depth_filled_linear = torch.from_numpy(depth_filled).float().unsqueeze(0)
            outputs["depth_filled_linear"] = depth_filled_linear
        else:
            outputs["depth_filled_linear"] = depth_raw_linear.clone()

        return outputs

    def _get_data_path(self, index):
        filename_line = self.filenames[index]

        # Get data path
        rgb_rel_path = filename_line[0]

        depth_rel_path, filled_rel_path = None, None
        # if DatasetMode.RGB_ONLY != self.mode:
        depth_rel_path = filename_line[1]
        if self.has_filled_depth:
            filled_rel_path = filename_line[2]
        return rgb_rel_path, depth_rel_path, filled_rel_path

    def _read_image(self, img_rel_path) -> np.ndarray:
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            image_to_read = self.tar_obj.extractfile("./" + img_rel_path)
            image_to_read = image_to_read.read()
            image_to_read = io.BytesIO(image_to_read)
        else:
            image_to_read = os.path.join(self.dataset_dir, img_rel_path)
        if self.read_event_via == 'default':
            image = Image.open(image_to_read)  # [H, W, rgb]
            image = np.asarray(image)
        elif self.read_event_via == 'tifffile': # for more than 3 channels
            image = tifffile.imread(image_to_read) # [N, H, W]
            image = np.transpose(image, (1, 2, 0)) # [H, W, N]
        else:
            raise NotImplementedError(f"Reading via {self.read_event_via} is not implemented.")
        # TODO: CHANGED! Maybe there's a better way to do this?
        # # If the image h and w are not divisible by 8, crop the image: crop the image by 8
        # factor = 8
        # if image.shape[0] % factor != 0 or image.shape[1] % factor != 0:
        #     image = image[: image.shape[0] // factor * factor, : image.shape[1] // factor * factor]
        return image

    def _read_rgb_file(self, rel_path) -> np.ndarray:
        rgb = self._read_image(rel_path)
        rgb = np.transpose(rgb, (2, 0, 1)).astype(int)  # [rgb, H, W]
        return rgb

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        #  Replace code below to decode depth according to dataset definition
        depth_decoded = depth_in

        return depth_decoded

    def _get_valid_mask(self, depth: torch.Tensor):
        valid_mask = torch.logical_and(
            torch.logical_and(depth > self.min_depth, depth < self.max_depth),
            ~torch.isnan(depth)
        ).bool()
        return valid_mask

    def _training_preprocess(self, rasters):
        # Augmentation
        if self.augm_args is not None and self.mode == DatasetMode.TRAIN:
            rasters = self._augment_data(rasters)

        if self.augm_args is not None and "random_crop_hw" in self.augm_args:
            hw = self.augm_args["random_crop_hw"]
            if self.mode == DatasetMode.TRAIN:
                rasters = self._random_crop(rasters, hw)
            else:
                rasters = self._center_crop(rasters, hw)


        # # debug
        # if torch.isnan(rasters["depth_raw_linear"]).any():
        #     print("[DEBUG] ⚠️ depth_raw_linear contains NaNs")

        # Normalization
        rasters["depth_raw_norm"] = self.depth_transform(
            rasters["depth_raw_linear"], rasters["valid_mask_raw"]
        ).clone()
        rasters["depth_filled_norm"] = self.depth_transform(
            rasters["depth_filled_linear"], rasters["valid_mask_filled"]
        ).clone()

        # # --- DEBUG: after normalization ---
        # # Quick NaN checks after normalization
        # if torch.isnan(rasters["depth_raw_norm"]).any():
        #     print("[DEBUG] ⚠️ depth_raw_norm contains NaNs")
        # if torch.isnan(rasters["depth_filled_norm"]).any():
        #     print("[DEBUG] ⚠️ depth_filled_norm contains NaNs")


        # Set invalid pixel to far plane
        if self.move_invalid_to_far_plane:
            if self.depth_transform.far_plane_at_max:
                rasters["depth_filled_norm"][~rasters["valid_mask_filled"]] = (
                    self.depth_transform.norm_max
                )
            else:
                rasters["depth_filled_norm"][~rasters["valid_mask_filled"]] = (
                    self.depth_transform.norm_min
                )

        # Resize
        if self.resize_to_hw is not None:
            resize_transform = Resize(
                size=self.resize_to_hw, interpolation=InterpolationMode.BILINEAR, antialias=True
            ) # originally nearest exact
            rasters = {k: resize_transform(v) for k, v in rasters.items()}

        return rasters

    def _augment_data(self, rasters_dict):
        # lr flipping
        lr_flip_p = self.augm_args.lr_flip_p
        if random.random() < lr_flip_p:
            rasters_dict = {k: v.flip(-1) for k, v in rasters_dict.items()}

        return rasters_dict

    def _random_crop(self, rasters: dict, crop_hw: tuple):
            """
            Randomly crop all tensors in rasters to (h, w),
            and record the absolute coordinates of each pixel in the crop.
            """
            crop_h, crop_w = crop_hw
            # pick a reference tensor to get H, W
            ref = next(iter(rasters.values()))
            _, H, W = ref.shape
            if H < crop_h or W < crop_w:
                raise ValueError(f"Crop size {crop_hw} larger than image {(H, W)}")

            # choose top‐left corner
            top = random.randint(0, H - crop_h)
            left = random.randint(0, W - crop_w)
            bottom = top + crop_h
            right = left + crop_w

            # build per‐pixel global_index = [[y_coords],[x_coords]] of shape [2, crop_h, crop_w]
            ys = torch.arange(top, bottom, dtype=torch.long)
            xs = torch.arange(left, right, dtype=torch.long)
            # meshgrid with “ij” indexing gives grid_y[i,j]=ys[i], grid_x[i,j]=xs[j]
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            global_index = torch.stack([grid_y, grid_x], dim=0)  # [2, Hc, Wc]

            # now actually crop every raster entry
            out = {}
            for k, v in rasters.items():
                # v is a tensor shaped [C, H, W]
                out[k] = v[..., top:bottom, left:right]

            # stash both the bounding box and the per‐pixel map
            out["crop_coords"] = torch.tensor([top, left, bottom, right], dtype=torch.long)
            out["global_index"] = global_index

            return out

    def _center_crop(self, rasters: dict, crop_hw: tuple):
        """Deterministic center‐crop + global_index like _random_crop does."""
        crop_h, crop_w = crop_hw
        ref = next(iter(rasters.values()))
        _, H, W = ref.shape
        if H < crop_h or W < crop_w:
            raise ValueError(f"Center crop {crop_hw} > image {(H, W)}")

        top  = (H - crop_h) // 2
        left = (W - crop_w) // 2
        bottom = top + crop_h
        right  = left + crop_w

        # build global_index same as random version
        ys = torch.arange(top,    bottom, dtype=torch.long)
        xs = torch.arange(left,   right,  dtype=torch.long)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        global_index = torch.stack([grid_y, grid_x], dim=0)  # [2,Hc,Wc]

        out = {}
        for k, v in rasters.items():
            out[k] = v[..., top:bottom, left:right]

        out["crop_coords"]  = torch.tensor([top, left, bottom, right])
        out["global_index"] = global_index
        return out


    def __del__(self):
        if hasattr(self, "tar_obj") and self.tar_obj is not None:
            self.tar_obj.close()
            self.tar_obj = None


def get_pred_name(rgb_basename, name_mode, suffix=".png"):
    if DepthFileNameMode.rgb_id == name_mode:
        pred_basename = "pred_" + rgb_basename.split("_")[1]
    elif DepthFileNameMode.i_d_rgb == name_mode:
        pred_basename = rgb_basename.replace("_rgb.", "_pred.")
    elif DepthFileNameMode.id == name_mode:
        pred_basename = "pred_" + rgb_basename
    elif DepthFileNameMode.rgb_i_d == name_mode:
        pred_basename = "pred_" + "_".join(rgb_basename.split("_")[1:])
    else:
        raise NotImplementedError
    # change suffix
    pred_basename = os.path.splitext(pred_basename)[0] + suffix

    return pred_basename


