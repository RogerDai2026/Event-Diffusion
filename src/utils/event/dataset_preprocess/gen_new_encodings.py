# #!/usr/bin/env python3
# import os
# import glob
# import argparse
# import numpy as np
# import re
# from numpy import ndarray
# from tqdm import tqdm
# import tifffile
#
#
# def normalize(x, relative_vmin=None, relative_vmax=None, interval_vmax=None):
#     vmax = x.max()
#     vmin = x.min()
#     if (relative_vmax is not None):
#         vmax = relative_vmax + vmin
#     if (relative_vmin is not None):
#         vmin = relative_vmin + vmin
#     if (interval_vmax is None):
#         interval_vmax = vmax - vmin
#
#
#     # Keep only the values between vmin and vmax
#     x = x * (x >= vmin) * (x <= vmax)
#
#     return (x - vmin) / interval_vmax
#
# # TIME_ENCODING = None
#
# def nbin_encoding(times: ndarray, polarity: ndarray, x: ndarray, y: ndarray, args: dict, nbin: int) -> ndarray:
#     x = x.astype(np.int64)
#     y = y.astype(np.int64)
#     normalized_time = normalize(times)
#     polarity[polarity == 1] = 255
#     polarity[polarity == -1] = 0
#     encoded_image = np.ones((nbin, args.height, args.width), dtype=np.int8) * 128
#     time_bin = np.minimum((normalized_time * nbin).astype(int), nbin - 1)
#     for b in range(nbin):
#         cur_frame = np.ones((args.height, args.width), dtype=np.int8) * 128
#         mask = (time_bin == b)
#         cur_frame[y[mask], x[mask]] = polarity[mask]
#         encoded_image[b] = cur_frame
#     return encoded_image
#
# def process_and_encode(
#     npy_path: str,
#     output_dir: str,
#     nbin: int
# ) -> None:
#     """
#     Load a raw event .npy (structured or unstructured), infer dimensions if needed,
#     apply nbin_encoding, and save the result as a TIFF stack.
#     """
#     data = np.load(npy_path)
#
#     # Structured event array: fields 't','x','y','pol'
#     if hasattr(data, 'dtype') and data.dtype.names is not None:
#         t = data['t'].astype(np.float64) / 1e6
#         x = data['x'].astype(np.int64)
#         y = data['y'].astype(np.int64)
#         p = data['pol'].astype(np.int8)
#         height = int(y.max()) + 1
#         width = int(x.max()) + 1
#     else:
#         # 3D voxel grid: (bins, H, W)
#         if data.ndim == 3:
#             encoded = data.astype(np.int8)
#             basename = os.path.splitext(os.path.basename(npy_path))[0]
#             out_name = f"{basename}_nbin{data.shape[0]}.tif"
#             out_path = os.path.join(output_dir, out_name)
#             tifffile.imwrite(out_path, encoded, bigtiff=True)
#             return
#         # Unstructured events Nx4
#         t = data[:, 0].astype(np.float64) / 1e6
#         x = data[:, 1].astype(np.int64)
#         y = data[:, 2].astype(np.int64)
#         p = data[:, 3].astype(np.int8)
#         height = int(y.max()) + 1
#         width = int(x.max()) + 1
#
#     # Create args namespace for encoder
#     class A: pass
#     args = A()
#     args.height = height
#     args.width = width
#
#     # Encode
#     encoded = nbin_encoding(t, p, x, y, args, nbin)
#     print(f"shape: {encoded.shape}")
#
#     # Save output preserving frame ID as TIFF
#     basename = os.path.splitext(os.path.basename(npy_path))[0]
#     out_name = f"{basename}_nbin{nbin}.tif"
#     out_path = os.path.join(output_dir, out_name)
#     tifffile.imwrite(out_path, encoded.astype(np.int8), bigtiff=True)
#
#
# def generate_pairs(
#     enc_dir: str,
#     depth_dir: str,
#     pair_file: str,
#     n_samples: int
# ) -> None:
#     """
#     Scan enc_dir for encoded TIFF files, match frame IDs with depth_dir (.npy),
#     and write pairs to pair_file.
#     """
#     enc_paths = sorted(glob.glob(os.path.join(enc_dir, '*_nbin*.tif')))[:n_samples]
#     with open(pair_file, 'w') as f:
#         for enc in enc_paths:
#             fname = os.path.basename(enc)
#             # Extract numeric frame ID
#             m = re.search(r'frame[_-]?(0*\d+)', fname)
#             if m:
#                 frame_id = m.group(1)
#             else:
#                 m2 = re.search(r'(\d+)', fname)
#                 frame_id = m2.group(1) if m2 else fname
#             # find matching depth .npy file
#             depth_pattern = os.path.join(depth_dir, f"*{frame_id}*.npy")
#             candidates = glob.glob(depth_pattern)
#             if not candidates:
#                 print(f"Warning: no depth for frame {frame_id}")
#                 continue
#             depth_path = candidates[0]
#             f.write(f"{enc} {depth_path}\n")
#     print(f"Wrote {len(enc_paths)} pairs to {pair_file}")
#
#
#
# def main(
#     input_dir: str,
#     output_dir: str,
#     depth_dir: str,
#     num_samples: int,
#     nbin: int
# ):
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 1) Encode events
#     files = sorted(glob.glob(os.path.join(input_dir, '*.npy')))
#     if not files:
#         raise RuntimeError(f"No .npy files found in {input_dir}")
#     selected = files[:num_samples]
#     print(f"Encoding {len(selected)} event files from {input_dir}")
#     for path in tqdm(selected, desc="Encoding events"):
#         try:
#             process_and_encode(path, output_dir, nbin)
#         except Exception as e:
#             print(f"Failed {path}: {e}")
#     print("Encoding complete.")
#
#     # 2) Generate pairs file
#     pair_file = os.path.join(output_dir, 'pairs.txt')
#     generate_pairs(output_dir, depth_dir, pair_file, num_samples)
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description="Encode event npy and produce event-depth pairs"
#     )
#     parser.add_argument(
#         '--input_dir', type=str,
#         default='/shared/ad150/event3d/MVSEC/mvsec_outdoor_day1/events/voxels/',
#         help='Directory with raw event .npy files'
#     )
#     parser.add_argument(
#         '--output_dir', type=str,
#         default='/shared/qd8/event3d/MVSEC/',
#         help='Directory to save encoded .npy outputs'
#     )
#     parser.add_argument(
#         '--depth_dir', type=str,
#         default='/shared/ad150/event3d/MVSEC/mvsec_outdoor_day1/depth/data/',
#         help='Directory with depth .png frames'
#     )
#     parser.add_argument(
#         '--num_samples', type=int, default=20,
#         help='Number of files to process and pair'
#     )
#     parser.add_argument(
#         '--nbin', type=int, default=5,
#         help='Number of time bins for encoding'
#     )
#     args = parser.parse_args()
#
#     main(
#         args.input_dir,
#         args.output_dir,
#         args.depth_dir,
#         args.num_samples,
#         args.nbin
#     )



#this is for generating encoding for one sample for mvsec
#!/usr/bin/env python3
#!/usr/bin/env python3
import os
import numpy as np
import tifffile
from numpy import ndarray
import matplotlib.pyplot as plt

def normalize(x: ndarray) -> ndarray:
    """Normalize timestamps to [0, 1]."""
    if x.size == 0:
        return x
    mn, mx = x.min(), x.max()
    if mx > mn:
        return (x - mn) / (mx - mn)
    else:
        return np.zeros_like(x)

def nbin_encoding(
    times: ndarray,
    polarity: ndarray,
    x: ndarray,
    y: ndarray,
    height: int,
    width: int,
    nbin: int
) -> ndarray:
    """
    Generate an (nbin, H, W) uint8 stack with values in {0,128,255}:
      - neutral baseline = 128
      - polarity +1 -> 255
      - polarity -1 ->   0
    """
    # cast coords
    x = x.astype(np.int64)
    y = y.astype(np.int64)

    # normalize timestamps to [0,1)
    tnorm = normalize(times)

    # make a 0/255 uint8 polarity array
    pol_u8 = np.where(polarity == 1, 255, 0).astype(np.uint8)

    # initialize with neutral gray = 128
    encoded = np.full((nbin, height, width), 128, dtype=np.uint8)

    # assign each event to its bin
    bins = np.minimum((tnorm * nbin).astype(int), nbin - 1)
    for b in range(nbin):
        mask = (bins == b)
        if not np.any(mask):
            continue
        frame = np.full((height, width), 128, dtype=np.uint8)
        frame[y[mask], x[mask]] = pol_u8[mask]
        encoded[b] = frame

    return encoded

def process_one(npy_path: str, depth_path: str, output_dir: str, nbin: int):
    # load events (either .npy or .npz)
    data = np.load(npy_path, allow_pickle=True)
    if isinstance(data, np.ndarray):
        arr = data
    else:
        # .npz case, pick first array
        arr = data[data.files[0]]
        # If batched (1,5,H,W), squeeze leading dim
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
        # Expect shape (5, H, W)
    nbin, H, W = arr.shape

    # Map from [-1,1] → [0,1] for display
    arr_disp = (arr + 1.0) / 2.0
    arr_disp = np.clip(arr_disp, 0.0, 1.0)

    # Plot each bin
    fig, axs = plt.subplots(1, nbin, figsize=(nbin * 3, 3))
    for i in range(nbin):
        ax = axs[i]
        im = ax.imshow(arr_disp[i], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Bin {i}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    if isinstance(data, np.ndarray):
        ev = data
    else:
        keys = data.files
        ev = data["events"] if "events" in keys else data[keys[0]]

    # unpack structured vs unstructured
    if hasattr(ev, "dtype") and ev.dtype.names is not None:
        t = ev["t"].astype(np.float64) / 1e6
        x = ev["x"].astype(np.int64)
        y = ev["y"].astype(np.int64)
        p = ev["pol"].astype(np.int8)
    else:
        arr = ev
        t = arr[:, 0].astype(np.float64) / 1e6
        x = arr[:, 1].astype(np.int64)
        y = arr[:, 2].astype(np.int64)
        p = arr[:, 3].astype(np.int8)

    # infer spatial dims
    H = int(y.max()) + 1
    W = int(x.max()) + 1

    # encode and print a quick sanity check
    encoded = nbin_encoding(t, p, x, y, H, W, nbin)
    print(f"[encode] {os.path.basename(npy_path)} -> shape {encoded.shape}")
    for b in range(encoded.shape[0]):
        corner = encoded[b, :3, :3]
        print(f"  bin {b} corner:\n{corner}")

    # save TIFF
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(npy_path))[0]
    tif_path = os.path.join(output_dir, f"{base}_nbin{nbin}.tif")
    tifffile.imwrite(tif_path, encoded, bigtiff=True)
    print(f"Saved: {tif_path}")

    # write pairs.txt
    pair_txt = os.path.join(output_dir, "pairs.txt")
    with open(pair_txt, "w") as f:
        f.write(f"{tif_path} {depth_path}\n")
    print(f"Wrote pairs: {pair_txt}\n")


if __name__ == "__main__":
    # Hardcoded defaults for immediate execution
    event_npy   = '/shared/ad150/event3d/MVSEC/mvsec_outdoor_day1/events/voxels/event_tensor_0000003562.npy'
    depth_img   = '/shared/ad150/event3d/MVSEC/mvsec_outdoor_day1/depth/frames/frame_0000003562.png'
    output_dir  = '/shared/qd8/event3d/MVSEC/'
    nbin        = 5

    process_one(
        npy_path   = event_npy,
        depth_path = depth_img,
        output_dir = output_dir,
        nbin       = nbin
    )
