# Last modified: 2024-02-08
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


from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode
import numpy as np
import tifffile
from PIL import Image, ImageFile
import os
import re
import torch


class MVSECDataset(BaseDepthDataset):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            # MVSEC data parameter
            min_depth=1e-5,
            max_depth=250, # 1000.0,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )

    def _get_data_path(self, index):
        """Override to handle custom file paths directly from filename pairs"""
        filename_line = self.filenames[index]
        # The filenames should be direct paths to event and depth files
        event_rel_path = filename_line[0]  # events/dvs_scene51_frame_003416.tiffle
        depth_rel_path = filename_line[1]  # depth/depth_scene51_frame_003418.png

        return event_rel_path, depth_rel_path, None

    def _extract_frame_number(self, filename):
        """Extract frame number from filename"""
        match = re.search(r'frame_(\d+)', filename)
        return int(match.group(1)) if match else None

    def _read_depth_file(self, rel_path):
        """
        Load a .npy or .npz depth file and return a float32 H×W numpy array.
        """
        full_path = os.path.join(self.dataset_dir, rel_path)
        loaded = np.load(full_path)

        # If .npz, grab the first array in the archive
        if isinstance(loaded, np.lib.npyio.NpzFile):
            key = next(iter(loaded.files))
            arr = loaded[key]
        else:
            arr = loaded

        # Ensure float32
        arr = arr.astype(np.float32)

        # If it has a singleton channel dimension at the end, squeeze it off
        # (e.g. shape (H, W, 1) → (H, W))
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.ndim != 2:
            raise ValueError(f"Expected 2D depth map or H×W×1, but got shape {arr.shape}")

        return arr


    # def _read_npy_file(self, rel_path):
    #     npy_path_or_content = os.path.join(self.dataset_dir, rel_path)
    #     image = np.load(npy_path_or_content).squeeze()
    #
    #     # TODO: CHANGED! Maybe there's a better way to do this?
    #     # If the image h and w are not divisible by 8, crop the image
    #     factor = 8
    #     if image.shape[0] % factor != 0 or image.shape[1] % factor != 0:
    #         image = image[: image.shape[0] // factor * factor, : image.shape[1] // factor * factor]
    #
    #     data = image[np.newaxis, :, :]
    #     return data

    def _read_rgb_file(self, rel_path) -> np.ndarray:
        """
        Now just load your pre-computed event 'image' (e.g. .tif or .png) as
        H×W×C float32 and return C×H×W.
        """
        full = os.path.join(self.dataset_dir, rel_path)
        # use tifffile for multi-channel, PIL for anything else
        if full.lower().endswith(('.tif', '.tiff')):
            img = tifffile.imread(full)  # shape: (C, H, W) or (H, W, C)
            # ensure HWC
            if img.ndim == 3 and img.shape[0] <= 4:
                # assume (C,H,W) → (H,W,C)
                # img = np.transpose(img, (1, 2, 0))
                pass
        elif full.lower().endswith(('.npz', '.npy')):
            data = np.load(full)
            if isinstance(data, np.lib.npyio.NpzFile):
                # pick the first array in the archive
                key = next(iter(data.files))
                arr = data[key]
            else:
                arr = data  # .npy → direct array

            img = arr.astype(np.float32)
            # now img is H×W or H×W×C
            if img.ndim == 2:
                # single‐channel → add channel dim
                img = img[:, :, None]
        else:
            img = np.asarray(Image.open(full))  # always H×W×C
        # make sure it's float32
        img = img.astype(np.float32)
        # transpose to CHW
        return img
