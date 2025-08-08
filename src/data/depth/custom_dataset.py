import os
import re
import glob
import random
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import tifffile
from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode


class CustomDataset(BaseDepthDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            # Custom data parameters
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
        """Read and convert depth from PNG RGB format"""
        depth_path = os.path.join(self.dataset_dir, rel_path)

        # Read the PNG image
        depth_img = Image.open(depth_path)
        depth_array = np.array(depth_img)

        # Convert from RGB to depth using the provided formula
        if len(depth_array.shape) == 3:  # RGB image
            R = depth_array[:, :, 0].astype(np.float64)
            G = depth_array[:, :, 1].astype(np.float64)
            B = depth_array[:, :, 2].astype(np.float64)

            # normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
            normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)

            # in_meters = 1000 * normalized
            depth_meters = 500.0 * normalized
        else:
            # If it's already grayscale, treat as normalized depth
            depth_meters = depth_array.astype(np.float64)


        # Add channel dimension
        return depth_meters[np.newaxis, :, :]



    # def _read_rgb_file(self, rel_path) -> np.ndarray:
    #     """Read event data and convert to RGB format"""
    #     file_path = os.path.join(self.dataset_dir, rel_path)
    #
    #     if rel_path.endswith('.tif') or rel_path.endswith('.tiff'):
    #         # Handle TIFF files using base class method
    #         rgb_int = super()._read_rgb_file(rel_path)
    #         # Cast to float32 here
    #         return rgb_int.astype(np.float32)
    #
    #     elif rel_path.endswith('.npz'):
    #         file_path = os.path.join(self.dataset_dir, rel_path)
    #
    #         if rel_path.endswith('.npz'):
    #             # ─── INSERT YOUR SIZE CHECK HERE ────────────────────────────────
    #             try:
    #                 data = np.load(file_path)
    #             except Exception as e:
    #                 raise ValueError(f"Could not load npz file {file_path}: {e}")
    #
    #         # Extract event data
    #         if 'events' in data:
    #             events = data['events']
    #         else:
    #             # Try the first available key
    #             available_keys = list(data.keys())
    #             events = data[available_keys[0]]
    #
    #         # Handle structured array (common DVS format)
    #         if events.dtype.names is not None:
    #             # Extract fields from structured array
    #             x = events['x'].astype(np.int32)
    #             y = events['y'].astype(np.int32)
    #             t = events['t'].astype(np.float32)
    #             pol = events['pol'].astype(np.int32)
    #
    #             # Determine image dimensions from coordinate ranges
    #             height = int(y.max()) + 1
    #             width = int(x.max()) + 1
    #
    #             # Create event representation using LINEAR time encoding
    #             # Normalize timestamps to [0, 1] range
    #             if len(t) > 0:
    #                 t_min, t_max = t.min(), t.max()
    #                 if t_max > t_min:
    #                     t_norm = (t - t_min) / (t_max - t_min)
    #                 else:
    #                     t_norm = np.zeros_like(t,dtype=np.float32)
    #             else:
    #                 t_norm = np.array([])
    #
    #             # Initialize image (3-channel RGB)
    #             event_img = np.zeros((height, width, 3), dtype=np.float32)
    #
    #             if len(events) > 0:
    #                 # Convert boolean polarity to -1/+1 if needed
    #                 if pol.dtype == bool:
    #                     pol = pol.astype(np.int32) * 2 - 1  # True->1, False->-1
    #                 elif np.all(np.isin(pol, [0, 1])):
    #                     pol = pol * 2 - 1  # 0->-1, 1->1
    #
    #                 # Split events by polarity
    #                 pos_mask = pol > 0
    #                 neg_mask = pol <= 0
    #
    #                 # Assign positive events to red channel, negative to blue channel
    #                 if np.any(pos_mask):
    #                     event_img[y[pos_mask], x[pos_mask], 0] = t_norm[pos_mask]  # Red channel
    #
    #                 if np.any(neg_mask):
    #                     event_img[y[neg_mask], x[neg_mask], 2] = t_norm[neg_mask]  # Blue channel
    #
    #                 # # Optional: create intensity representation in green channel
    #                 # # Count events per pixel for intensity
    #                 # for i in range(len(x)):
    #                 #     event_img[y[i], x[i], 1] = min(event_img[y[i], x[i], 1] + 0.1, 1.0)
    #
    #         else:
    #             # Handle regular array format (fallback)
    #             if len(events.shape) == 2 and events.shape[1] >= 4:
    #                 # Assume [t, x, y, p] or [x, y, t, p] format
    #                 t = events[:, 0].astype(np.float32)
    #                 x = events[:, 1].astype(np.int32)
    #                 y = events[:, 2].astype(np.int32)
    #                 pol = events[:, 3].astype(np.int32)
    #
    #                 height = int(y.max()) + 1
    #                 width = int(x.max()) + 1
    #
    #                 # Create simple binary event image
    #                 event_img = np.zeros((height, width, 3), dtype=np.float32)
    #                 event_img[y, x, :] = 1.0
    #             else:
    #                 # Create placeholder image for unknown format
    #                 raise NotImplementedError("image for unknown format")
    #
    #         # Convert to uint8 and scale to [0, 255]
    #         event_img = (event_img * 255).astype(np.uint8)
    #
    #         # # Apply cropping to make divisible by 8
    #         # factor = 8
    #         # if event_img.shape[0] % factor != 0 or event_img.shape[1] % factor != 0:
    #         #     event_img = event_img[
    #         #         : event_img.shape[0] // factor,
    #         #         : event_img.shape[1] // factor
    #         #     ]
    #         #
    #         # Convert to [C, H, W] format like base class expects
    #         rgb = np.transpose(event_img, (2, 0, 1)).astype(int)  # [C, H, W]
    #         return rgb
    #
    #     else:
    #         # For other file types, use base class method
    #         rgb_int = super()._read_rgb_file(rel_path)
    #         # Cast to float32
    #         return rgb_int.astype(np.float32)
    #
    def _read_rgb_file(self, rel_path) -> np.ndarray:
        """
        Now just load your pre-computed event 'image' (e.g. .tif or .png) as
        H×W×C float32 and return C×H×W.
        """
        full = os.path.join(self.dataset_dir, rel_path)
        # use tifffile for multi-channel, PIL for anything else
        if full.lower().endswith(('.tif', '.tiff')):
            img = tifffile.imread(full)        # shape: (C, H, W) or (H, W, C)
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
