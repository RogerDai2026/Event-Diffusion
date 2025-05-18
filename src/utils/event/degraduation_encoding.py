import torch.nn.functional as F
import math
import torch
import matplotlib.pyplot as plt
from skimage.draw import polygon
import numpy as np


def get_gaussian_kernel(kernel_size: int, sigma: float, plot_psf: bool = False) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel.
    """
    ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size // 2)
    gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
    gauss = gauss / gauss.sum()
    kernel = torch.outer(gauss, gauss)
    if plot_psf:
        plt.imshow(kernel.numpy(), cmap='viridis', origin='lower')
        plt.title(f"Gaussian Kernel (PSF) - size={kernel_size}, sigma={sigma}")
        plt.colorbar()
        plt.show()

    return kernel

def gaussian_blur_encoding(image, kernel_size: int, sigma: float) -> torch.Tensor:
    """
    Apply Gaussian blur encoding on the input image while preserving its original dimensions.

    Args:
        image: torch.Tensor of shape (H, W) for grayscale or (C, H, W) for multi-channel images.
        kernel_size: Size of the Gaussian kernel (should be odd).
        sigma: Standard deviation of the Gaussian.

    Returns:
        Blurred image with the same shape as input.
    """
    # Record the original dimensions.
    original_dim = image.dim()  # 2 for grayscale (H,W) or 3 for (C,H,W)

    # Convert to 4D tensor (B, C, H, W)
    if original_dim == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # becomes (1, 1, H, W)
    elif original_dim == 3:
        image = image.unsqueeze(0)  # becomes (1, C, H, W)

    channels = image.shape[1]
    kernel = get_gaussian_kernel(kernel_size, sigma, plot_psf = True).to(image.device)
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    blurred = F.conv2d(image, kernel, groups=channels, padding=padding)

    # Remove extra dimensions to return to original shape
    if original_dim == 2:
        blurred = blurred.squeeze(0).squeeze(0)  # returns (H, W)
    elif original_dim == 3:
        blurred = blurred.squeeze(0)  # returns (C, H, W)

    return blurred



def regular_polygon_vertices(sides: int, radius: float = 0.5, angle_offset: float = 0.0):
    vertices = []
    for i in range(sides):
        angle = angle_offset + 2 * math.pi * i / sides
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        vertices.append((x, y))
    return vertices


# def point_in_polygon(x: float, y: float, vertices: list) -> bool:
#     inside = False
#     n = len(vertices)
#     for i in range(n):
#         x1, y1 = vertices[i]
#         x2, y2 = vertices[(i + 1) % n]
#         intersects = ((y1 > y) != (y2 > y))
#         if intersects:
#             x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
#             if x_intersect > x:
#                 inside = not inside
#     return inside

#
# def draw_line(mask: torch.Tensor, x0: float, y0: float, x1: float, y1: float,
#               coords: torch.Tensor, thickness: int = 1):
#     steps = 4 * mask.shape[0]
#     xs = torch.linspace(x0, x1, steps)
#     ys = torch.linspace(y0, y1, steps)
#     N = mask.shape[0]
#     for i in range(steps):
#         x_val = xs[i]
#         y_val = ys[i]
#         col = torch.argmin(torch.abs(coords - x_val))
#         row = torch.argmin(torch.abs(coords - y_val))
#         if 0 <= row < N and 0 <= col < N:
#             for dr in range(-thickness + 1, thickness):
#                 for dc in range(-thickness + 1, thickness):
#                     rr = row + dr
#                     cc = col + dc
#                     if 0 <= rr < N and 0 <= cc < N:
#                         mask[rr, cc] = 1.0
#
#
# def line_box_intersections(x1, y1, x2, y2, box_min=-1.0, box_max=1.0):
#     EPS = 1e-9
#     points = []
#     dx = x2 - x1
#     dy = y2 - y1
#
#     def check_intersection(t):
#         if t is not None:
#             ix = x1 + t * dx
#             iy = y1 + t * dy
#             if box_min - EPS <= ix <= box_max + EPS and box_min - EPS <= iy <= box_max + EPS:
#                 points.append(((ix, iy), t))
#
#     if abs(dx) > EPS:
#         t = (box_min - x1) / dx
#         check_intersection(t)
#     if abs(dx) > EPS:
#         t = (box_max - x1) / dx
#         check_intersection(t)
#     if abs(dy) > EPS:
#         t = (box_min - y1) / dy
#         check_intersection(t)
#     if abs(dy) > EPS:
#         t = (box_max - y1) / dy
#         check_intersection(t)
#     unique_pts = []
#     for (pt, tval) in points:
#         if pt not in [p for (p, _) in unique_pts]:
#             unique_pts.append((pt, tval))
#     return unique_pts

#
# def create_aperture_one_sided_edges(N=256, sides=6, polygon_radius=0.3,
#                                     angle_offset=0.0, blade_thickness=1):
#     mask = torch.zeros((N, N), dtype=torch.float32)
#     coords = torch.linspace(-1, 1, N)
#     vertices = regular_polygon_vertices(sides, polygon_radius, angle_offset)
#     for row in range(N):
#         y = coords[row]
#         for col in range(N):
#             x = coords[col]
#             if point_in_polygon(x, y, vertices):
#                 mask[row, col] = 1.0
#     # for i in range(len(vertices)):
#     #     x1, y1 = vertices[i]
#     #     x2, y2 = vertices[(i + 1) % len(vertices)]
#     #     dx = x2 - x1
#     #     dy = y2 - y1
#     #     box_pts = line_box_intersections(x1, y1, x2, y2, box_min=-1, box_max=1)
#     #     if not box_pts:
#     #         continue
#     #     valid_t = []
#     #     for (pt, tval) in box_pts:
#     #         if tval > 1.0:
#     #             valid_t.append((pt, tval))
#     #     if not valid_t:
#     #         continue
#     #     valid_t.sort(key=lambda x: x[1])
#     #     (ix, iy), t_ext = valid_t[0]
#     #     x_edge_end = x1 + 1.0 * dx
#     #     y_edge_end = y1 + 1.0 * dy
#     #     draw_line(mask, x_edge_end, y_edge_end, ix, iy, coords, thickness=blade_thickness)
#     return mask


def create_aperture_one_sided_edges(N=256, sides=6, polygon_radius=0.3,
                                    angle_offset=0.0, blade_thickness=1):
    verts = regular_polygon_vertices(sides, polygon_radius, angle_offset)
    # map (x,y)∈[–1,1] to image coords [0,N)
    pts = np.array([[(x+1)*(N-1)/2, (1-y)*(N-1)/2] for x,y in verts])
    rr,cc = polygon(pts[:,1], pts[:,0], shape=(N,N))
    mask = np.zeros((N,N), bool)
    mask[rr,cc] = True
    return torch.from_numpy(mask)


def compute_psf(aperture_mask: torch.Tensor):
    # Compute the 2D Fourier Transform to obtain the PSF.
    # PSF is the squared magnitude of the FFT.
    ft = torch.fft.fft2(aperture_mask)
    ft_shifted = torch.fft.fftshift(ft)
    psf = torch.abs(ft_shifted) ** 2
    psf = psf / psf.max()  # Normalize
    psf_log = torch.log10(psf + 1e-6)  # Use log-scale for better visualization

    # Convert to a NumPy array with float dtype
    psf_np = psf_log.cpu().detach().numpy()

    # Return the computed PSF
    return psf_np
if __name__ == "__main__":
    # Generate the aperture (sunstar simulation)
    N = 512
    sides = 6
    aperture_mask = create_aperture_one_sided_edges(
        N=N,
        sides=sides,
        polygon_radius=0.3,
        angle_offset=0.0,
        blade_thickness=1
    )

    # Compute the PSF from the aperture
    psf = compute_psf(aperture_mask)

    # Plot the aperture mask
    plt.figure(figsize=(6, 6))
    plt.imshow(aperture_mask.numpy(), origin='lower', cmap='gray')
    plt.title(f"Aperture Mask ({sides}-sided Polygon, One-Sided Edges)")
    plt.axis('off')
    plt.show()

    # Plot the PSF (sunstar diffraction pattern)
    plt.figure(figsize=(6, 6))
    plt.imshow(psf, origin='lower', cmap='inferno', extent=[-1, 1, -1, 1])
    plt.title("Sunstar PSF (Diffraction Pattern)")
    plt.xlabel("Spatial Frequency")
    plt.ylabel("Spatial Frequency")
    plt.colorbar(label='Normalized Intensity')
    plt.show()

# if __name__ == "__main__":
#     print("Current working directory:", os.getcwd())
#     print("hello")
#     # Create a dummy grayscale image of shape (128, 128)
#     image_path = "/home/qdai/Public/WechatIMG215.jpg"
#     img = Image.open(image_path)
#
#     # If you want to work with a grayscale image, convert it:
#     img = img.convert("L")  # "L" mode means grayscale
#
#     # Convert the PIL image to a PyTorch tensor
#     # For grayscale images, ToTensor() returns a tensor with shape (1, H, W)
#     transform = transforms.ToTensor()
#     img_tensor = transform(img).squeeze(0)  # Remove the channel dimension to get shape (H, W)
#
#     # Now you can test your filters:
#     blurred_image = gaussian_blur_encoding(img_tensor, kernel_size=11, sigma=6.0)
#
#     # Plot the images
#     plt.figure(figsize=(12, 4))
#
#     plt.subplot(1, 3, 1)
#     plt.imshow(img_tensor, cmap="gray")
#     plt.title("Original Image")
#     plt.axis("off")
#
#     plt.subplot(1, 3, 2)
#     plt.imshow(blurred_image, cmap="gray")
#     plt.title("Gaussian Blur")
#     plt.axis("off")
#
#     plt.subplot(1, 3, 3)
#     plt.imshow(sunstar_image, cmap="gray")
#     plt.title("Sunstar Filter")
#     plt.axis("off")
#
#     plt.tight_layout()
#     plt.savefig("filtered_images.png")  # Save the figure as an image file
#     print("Saved combined image as filtered_images.png")