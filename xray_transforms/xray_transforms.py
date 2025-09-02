import random
# from mmdet.registry import TRANSFORMS

from typing import Tuple, Union
import albumentations as A
from .neglog import *
from .window import *
from .gaussian_contrast import *
from .dropout import *
from .intensity_transforms import *

import pdb
import json
import os.path as osp

def build_augmentation(img: np.ndarray) -> np.ndarray:
    '''
    Required keys:
        - img
        
    Modified keys:
        - img
        
    '''
    # img = results["img"]
    
    # kwargs = dict(
    #     bbox_params=A.BboxParams(
    #         format="coco", label_fields=["category_ids"], min_visibility=0.1, min_area=10
    #     ),
    # )
    transforms =  A.Compose(
        [
            # neglog(),
            A.InvertImg(always_apply=True),
            A.OneOf(
                [
                    adjustable_window(0.1, 1, quantile=True),
                    # adjustable_window(0.1, 0.9, quantile=True),
                    mixture_window(keep_original=True, model="kmeans"),
                ],
                p=1.0,
            ),
            A.InvertImg(p=0.5),
            clahe(p=0.3),
            get_intensity_transforms(),
            # A.OneOf(
            #     [
            #         get_intensity_transforms(p=0.9),
            #         clahe(p=0.1),
            #     ],
            #     p=1,
            # ),
        ],
        # **kwargs,
    )
    
    # Apply the transformation and record replay data
    transforms = transforms(image=img)
    augmented_img = transforms["image"]
    
    # shuffle the channels
    if random.random() < 0.5:
        augmented_img = augmented_img[..., [1, 2, 0]]

    # results["img"] = augmented_img

    return augmented_img

def build_augmentation_val(img: np.ndarray) -> np.ndarray:
    '''
    Required keys:
        - img
        
    Modified keys:
        - img
        
    '''
    # img = results["img"]
    
    kwargs = dict(
        bbox_params=A.BboxParams(
            format="coco", label_fields=["category_ids"], min_visibility=0.1, min_area=10
        ),
    )
    transforms =  A.Compose(
        [
            # neglog(),
            A.InvertImg(always_apply=True),
            mixture_window(keep_original=True, model="kmeans"),
            A.InvertImg(p=0.5),
            clahe(p=0.3),
        ],
        # **kwargs,
    )
        
    # Apply the transformation and record replay data
    transforms = transforms(image=img)
    augmented_img = transforms["image"]
    
    # shuffle the channels
    if random.random() < 0.5:
        augmented_img = augmented_img[..., [1, 2, 0]]

    # results["img"] = augmented_img
    return augmented_img

def build_augmentation_real_xrays(results: dict) -> dict:
    '''
    Required keys:
        - img
        
    Modified keys:
        - img
        
    '''
    img = results["img"]
    
    transforms =  A.Compose(
        [
            neglog_lin_interp(),
            A.MedianBlur(blur_limit=5),
            mixture_window(keep_original=True, model="kmeans"),
            clahe(p=0.3),
        ],
        # **kwargs,
    )
        
    # Apply the transformation and record replay data
    transforms = transforms(image=img)
    augmented_img = transforms["image"]

    results["img"] = augmented_img
    return results
    

# @TRANSFORMS.register_module()
# class XrayTransforms:
#     def __init__(self, augmentation_fn=None):
#         """
#         Args:
#             augmentation_fn (callable, optional): Custom augmentation function.
#                 If None, defaults to `build_augmentation`.
#         """
#         self.augmentation = augmentation_fn or build_augmentation

#     def __call__(self, results: dict) -> dict:
#         """
#         Apply the augmentation to the results.

#         Args:
#             results (dict): Result dict containing the data to transform.

#         Returns:
#             dict: Augmented result dict.
#         """
#         return self.augmentation(results)

#     def __repr__(self) -> str:
#         """String representation for debugging."""
#         return f'{self.__class__.__name__}(augmentation_fn={self.augmentation})'
    
    
# @TRANSFORMS.register_module()
# class XrayTransforms_val:
#     def __init__(self, augmentation_fn=None):
#         """
#         Args:
#             augmentation_fn (callable, optional): Custom augmentation function.
#                 If None, defaults to `build_augmentation`.
#         """
#         self.augmentation = augmentation_fn or build_augmentation_val

#     def __call__(self, results: dict) -> dict:
#         """
#         Apply the augmentation to the results.

#         Args:
#             results (dict): Result dict containing the data to transform.

#         Returns:
#             dict: Augmented result dict.
#         """
#         return self.augmentation(results)

#     def __repr__(self) -> str:
#         """String representation for debugging."""
#         return f'{self.__class__.__name__}(augmentation_fn={self.augmentation})'
    
    
# @TRANSFORMS.register_module()
# class XrayTransforms_real_xrays:
#     def __init__(self, augmentation_fn=None):
#         """
#         Args:
#             augmentation_fn (callable, optional): Custom augmentation function.
#                 If None, defaults to `build_augmentation`.
#         """
#         self.augmentation = augmentation_fn or build_augmentation_real_xrays

#     def __call__(self, results: dict) -> dict:
#         """
#         Apply the augmentation to the results.

#         Args:
#             results (dict): Result dict containing the data to transform.

#         Returns:
#             dict: Augmented result dict.
#         """
#         return self.augmentation(results)

#     def __repr__(self) -> str:
#         """String representation for debugging."""
#         return f'{self.__class__.__name__}(augmentation_fn={self.augmentation})'