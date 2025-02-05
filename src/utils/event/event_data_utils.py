import torch
import torch.nn.functional as F
import math

def resize_to_multiple_of_16(x, mode='bilinear', align_corners=False):
    """
    Resizes the input tensor x so that its height and width are multiples of 16.
    This function uses interpolation to scale the input.

    Parameters:
        x (torch.Tensor): Input tensor of shape [N, C, H, W].
        mode (str): Interpolation mode (e.g., 'bilinear', 'nearest').
                    The default is 'bilinear'.
        align_corners (bool): Parameter for interpolation (only relevant for some modes).
                              Default is False.

    Returns:
        torch.Tensor: Resized tensor with height and width as multiples of 16.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    # Extract current spatial dimensions
    _, _, h, w = x.shape

    # Compute the next multiples of 16 for height and width
    new_h = int(math.ceil(h / 16.0) * 16)
    new_w = int(math.ceil(w / 16.0) * 16)

    # Use interpolate to resize the tensor to the new dimensions
    x_resized = F.interpolate(x, size=(new_h, new_w), mode=mode, align_corners=align_corners)
    return x_resized