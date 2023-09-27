import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/max(ma-mi, 1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_


def overlay_images(image_A, image_B, transparency):
    """
    Overlay two images with adjustable transparency.

    Args:
    - image_A (PIL Image): The base image.
    - image_B (PIL Image): The overlay image.
    - transparency (float): The transparency level (0.0 to 1.0, where 0.0 is fully transparent, and 1.0 is fully opaque).

    Returns:
    - PIL Image: The overlayed image.
    """
    # Ensure that both images have the same size
    if image_A.size != image_B.size:
        raise ValueError("Both input images must have the same size")

    # Convert the transparency value to an alpha value (0 to 255)
    alpha = int(255 * (1 - transparency))

    # Create copies of the images to avoid modifying the originals
    base_image = image_A.copy()
    overlay_image = image_B.copy()

    # Apply the transparency to the overlay image
    overlay_image.putalpha(alpha)

    # Paste the overlay image onto the base image
    base_image.paste(overlay_image, (0, 0), overlay_image)

    return base_image