import numpy as np
import albumentations as A

import logging
log = logging.getLogger(__name__)

def neglog_fn(images: np.ndarray, epsilon: float = 0.001) -> np.ndarray:
    """Take the negative log transform of an intensity image.

    Args:
        image (np.ndarray): [H,W,C] array of intensity images.
        epsilon (float, optional): positive offset from 0 before taking the logarithm.

    Returns:
        np.ndarray: the image or images after a negative log transform.
    """

    # shift image to avoid invalid values
    images = images.astype(np.float32)
    images += images.min(axis=(0, 1), keepdims=True) + epsilon

    # negative log transform
    images = -np.log(images)

    return images


def neglog(epsilon: float = 0.001) -> A.Lambda:
    """Take the negative log transform of an intensity image.

    Args:
    """

    def f_image(images: np.ndarray, **kwargs) -> np.ndarray:
        return neglog_fn(images, epsilon)

    def f_id(x, **kwargs):
        return x

    return A.Lambda(
        image=f_image,
        mask=f_id,
        bboxes=f_id,
        name="neglog",
    )

def neglog_lin_interp(epsilon: float = 0.001) -> A.Lambda:
    """Take the negative log transform of an intensity image and then it 
    performs linear interpolation to return in the range [0, 1]. This is the 
    same transformation done in DRR simulation when neglog is set to True when 
    projecting the image.

    Args:
    """

    def f_image(images: np.ndarray, **kwargs) -> np.ndarray:
        # apply neglog
        inv_img = neglog_fn(images, epsilon)
        # linearly interpolate to [0, 1]
        img_min = inv_img.min(axis=(0, 1), keepdims=True)
        img_max = inv_img.max(axis=(0, 1), keepdims=True)
        image = (
            inv_img - img_min
        ) / (img_max - img_min)
        return image

    def f_id(x, **kwargs):
        return x

    return A.Lambda(
        image=f_image,
        mask=f_id,
        bboxes=f_id,
        name="neglog",
    )