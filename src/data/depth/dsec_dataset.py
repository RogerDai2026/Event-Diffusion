import os
import re
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import tifffile
from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode


class DSECDataset(BaseDepthDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            # DSEC data parameters
            min_depth=0.1,      # Minimum depth in meters
            max_depth=100.0,    # Maximum depth in meters  
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )

    def _get_data_path(self, index):
        """Override to handle DSEC file paths directly from filename pairs"""
        filename_line = self.filenames[index]
        # The filenames should be direct paths to NBIN and depth files
        event_rel_path = filename_line[0]  # nbin_3_encoding/sequence/0000000000.tif
        depth_rel_path = filename_line[1]  # disparity_maps/sequence/disparity/image/000000.png
        
        return event_rel_path, depth_rel_path, None
    
    def _extract_frame_number(self, filename):
        """Extract frame number from DSEC filename"""
        # Handle both NBIN files (0000000000.tif) and depth files (000000.png)
        match = re.search(r'(\d{6,10})', filename)
        return int(match.group(1)) if match else None
    
    def _read_depth_file(self, rel_path):
        """Read DSEC disparity maps and convert to depth"""
        depth_path = os.path.join(self.dataset_dir, rel_path)

        # Read the 16-bit PNG disparity image
        depth_img = Image.open(depth_path)
        depth_array = np.array(depth_img)

        # DSEC format: 16-bit PNG disparity maps
        # According to DSEC docs:
        # - disp[y,x] = ((float)I[y,x])/256.0
        # - valid[y,x] = I[y,x]>0
        # - A value of 0 indicates invalid pixel
        
        if depth_array.dtype == np.uint16:
            # Convert to disparity values
            disparity = depth_array.astype(np.float32) / 256.0
            
            # Convert disparity to depth using camera parameters
            # DSEC stereo camera parameters (from calibration files):
            # - Baseline: 0.599033m (599mm)
            # - Focal length: 569.287 pixels (for rectified event cameras)
            baseline_focal = 341.022  # baseline * focal_length from cam_to_cam.yaml
            
            # Convert disparity to depth: depth = baseline * focal_length / disparity
            # Avoid division by zero by setting a minimum disparity
            disparity_safe = np.maximum(disparity, 0.01)  # Minimum disparity to avoid inf
            depth_meters = baseline_focal / disparity_safe
            
            # Set invalid pixels (original zeros) back to 0 - base class will handle masking
            depth_meters[depth_array == 0] = 0.0
                 
        else:
            # Fallback for other formats
            print(f"Warning: Unexpected depth format {depth_array.dtype} for {rel_path}")
            depth_meters = depth_array.astype(np.float32)

        # Add channel dimension [1, H, W]
        return depth_meters[np.newaxis, :, :]

    def _read_rgb_file(self, rel_path) -> np.ndarray:
        """
        Read DSEC NBIN encoded event data from TIFF files.
        NBIN files are pre-computed event representations in TIFF format.
        """
        full_path = os.path.join(self.dataset_dir, rel_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Event file not found: {full_path}")
        
        # Read TIFF file (NBIN encoded events)
        if full_path.lower().endswith(('.tif', '.tiff')):
            img = tifffile.imread(full_path)  # Shape: (C, H, W) or (H, W, C)
            
            # Ensure correct format
            if img.ndim == 3:
                if img.shape[0] <= 10:  # Assume (C, H, W) format
                    # Already in CHW format
                    pass
                else:  # Assume (H, W, C) format
                    img = np.transpose(img, (2, 0, 1))  # Convert to CHW
            elif img.ndim == 2:
                # Single channel, add channel dimension
                img = img[np.newaxis, :, :]
            else:
                raise ValueError(f"Unexpected image dimensions: {img.shape}")
                
        elif full_path.lower().endswith(('.npz', '.npy')):
            # Handle NPZ/NPY files if needed
            data = np.load(full_path)
            if isinstance(data, np.lib.npyio.NpzFile):
                # Pick the first array in the archive
                key = next(iter(data.files))
                img = data[key]
            else:
                img = data  # .npy → direct array

            # Ensure CHW format
            if img.ndim == 2:
                img = img[np.newaxis, :, :]  # Add channel dimension
            elif img.ndim == 3 and img.shape[-1] <= 10:
                img = np.transpose(img, (2, 0, 1))  # HWC to CHW
                
        else:
            # Fallback to PIL for other formats
            img = np.asarray(Image.open(full_path))
            if img.ndim == 2:
                img = img[np.newaxis, :, :]
            elif img.ndim == 3:
                img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        
        # Ensure float32 dtype
        img = img.astype(np.float32)

        return img

    def get_camera_info(self):
        """
        Return DSEC camera information from calibration files.
        These are the actual values from cam_to_cam.yaml.
        """
        return {
            'baseline': 0.599033,  # meters (from disparity_to_depth matrix)
            'focal_length': 569.287,  # pixels (rectified event cameras)
            'baseline_focal': 341.022,  # baseline * focal_length
            'image_width': 1440,  # frame camera resolution
            'image_height': 1080,
            'event_width': 640,   # event camera resolution
            'event_height': 480,
        }
