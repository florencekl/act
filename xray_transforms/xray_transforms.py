import random
# from mmdet.registry import TRANSFORMS
from PIL import Image

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

def build_augmentation(img: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    '''
    Required keys:
        - img
        
    Modified keys:
        - img
        
    '''
    # img = results["img"]
    
    # Resize image if it doesn't match target size
    if target_size is not None and img.shape[:2] != target_size:
        pil_img = Image.fromarray(img)
        img = np.array(pil_img.resize(target_size, Image.BILINEAR))
    
    # # Convert to 3-channel if needed (for normal backbone)
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = np.array([img, img, img]).transpose(1, 2, 0) if len(img.shape) == 2 else np.repeat(img, 3, axis=2)
    
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
                p=0.9,
            ),
            A.InvertImg(p=0.5),
            A.CLAHE(clip_limit=(4, 6), tile_grid_size=(8, 12), p=0.3),
            A.OneOf(
                [
                    get_intensity_transforms(),
                ],
                p=0.9,
            ),
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

    # print(np.shape(augmented_img), np.min(augmented_img), np.max(augmented_img))
    
    augmented_img = augmented_img[:,:,0]

    augmented_img = np.array([augmented_img, augmented_img, augmented_img]).transpose(1, 2, 0) if len(augmented_img.shape) == 2 else np.repeat(augmented_img, 3, axis=2)
    # results["img"] = augmented_img
    return augmented_img

def build_augmentation_val(img: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    '''
    Required keys:
        - img
        
    Modified keys:
        - img
        
    '''
    # img = results["img"]
    
    # Resize image if it doesn't match target size
    if target_size is not None and img.shape[:2] != target_size:
        pil_img = Image.fromarray(img)
        img = np.array(pil_img.resize(target_size, Image.BILINEAR))
    
    # # Convert to 3-channel if needed (for normal backbone)
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = np.array([img, img, img]).transpose(1, 2, 0) if len(img.shape) == 2 else np.repeat(img, 3, axis=2)

    transforms =  A.Compose(
        [
            # neglog(),
            A.InvertImg(always_apply=True),
            A.OneOf(
                [
                    mixture_window(keep_original=True, model="kmeans"),
                ],
                p=0.9,
            ),
            A.InvertImg(p=0.5),
            A.CLAHE(clip_limit=(4, 6), tile_grid_size=(8, 12), p=0.3),
        ],
        # **kwargs,
    )
        
    # Apply the transformation and record replay data
    transforms = transforms(image=img)
    augmented_img = transforms["image"]
    
    # shuffle the channels
    if random.random() < 0.5:
        augmented_img = augmented_img[..., [1, 2, 0]]

    augmented_img = augmented_img[:,:,0]

    augmented_img = np.array([augmented_img, augmented_img, augmented_img]).transpose(1, 2, 0) if len(augmented_img.shape) == 2 else np.repeat(augmented_img, 3, axis=2)
    # results["img"] = augmented_img
    return augmented_img

def build_replay_augmentation_val(img: np.ndarray, target_size: Tuple[int, int] = (256, 256), replay = None, lambda_transforms = None):
    '''
    Required keys:
        - img
        - replay
        
    Modified keys:
        - img
    '''

    transforms =  A.ReplayCompose(
        [
            # neglog(),
            A.InvertImg(always_apply=True),
            mixture_window(keep_original=True, model="kmeans"),
            A.InvertImg(p=0.5),
            clahe(p=0.3),
        ],
    )
    
    # Resize image if it doesn't match target size
    if target_size is not None and img.shape[:2] != target_size:
        pil_img = Image.fromarray(img)
        img = np.array(pil_img.resize(target_size, Image.BILINEAR))
    
    # Convert to 3-channel if needed (for normal backbone)
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = np.array([img, img, img]).transpose(1, 2, 0) if len(img.shape) == 2 else np.repeat(img, 3, axis=2)

    if replay is None:
        data = transforms(image=img)
        replay = data['replay']
        lambda_transforms = {lam.name: lam for lam in transforms if isinstance(lam, A.Lambda)}
        augmented_image = data["image"]

        # augs = transforms._restore_for_replay(replay, lambda_transforms=lambda_transforms)
        # data = augs(force_apply=True, image=img)
        # augmented_image = data["image"]

        # Create lambda_transforms dictionary to handle non-serializable transforms
        # augmented_image = transforms.replay(image=img, replay=replay)
        # **replay['replay']
    else:
        augs = transforms._restore_for_replay(replay, lambda_transforms=lambda_transforms)
        data = augs(force_apply=True, image=img)
        augmented_image = data["image"]

    # results["img"] = augmented_img
    return augmented_image, replay, lambda_transforms

def build_replay_augmentation_val(img: np.ndarray, target_size: Tuple[int, int] = (256, 256), replay = None, lambda_transforms = None, apply=True):
    '''
    Required keys:
        - img
        - replay
        
    Modified keys:
        - img
    '''

    transforms =  A.ReplayCompose(
        [
            # neglog(),
            A.InvertImg(always_apply=True),
            mixture_window(keep_original=True, model="kmeans"),
            A.InvertImg(p=0.5),
            clahe(p=0.3),
        ],
    )
    
    # Resize image if it doesn't match target size
    if target_size is not None and img.shape[:2] != target_size:
        pil_img = Image.fromarray(img)
        img = np.array(pil_img.resize(target_size, Image.BILINEAR))
    
    # Convert to 3-channel if needed (for normal backbone)
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = np.array([img, img, img]).transpose(1, 2, 0) if len(img.shape) == 2 else np.repeat(img, 3, axis=2)

    if apply:
        if replay is None:
            data = transforms(image=img)
            replay = data['replay']
            lambda_transforms = {lam.name: lam for lam in transforms if isinstance(lam, A.Lambda)}
            augmented_image = data["image"]

            # augs = transforms._restore_for_replay(replay, lambda_transforms=lambda_transforms)
            # data = augs(force_apply=True, image=img)
            # augmented_image = data["image"]

            # Create lambda_transforms dictionary to handle non-serializable transforms
            # augmented_image = transforms.replay(image=img, replay=replay)
            # **replay['replay']
        else:
            augs = transforms._restore_for_replay(replay, lambda_transforms=lambda_transforms)
            data = augs(force_apply=True, image=img)
            augmented_image = data["image"]
    else:
        augmented_image = img

    # results["img"] = augmented_img
    # augmented_image = img
    # print(np.shape(augmented_image), print(np.min(augmented_image), np.max(augmented_image)))
    # print(np.shape(img), print(np.min(img), np.max(img)))
    return augmented_image, replay, lambda_transforms

def build_augmentation_real_xrays(img: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> dict:
    '''
    Required keys:
        - img
        
    Modified keys:
        - img
        
    '''
    # img = results["img"]

    if target_size is not None and img.shape[:2] != target_size:
        pil_img = Image.fromarray(img)
        img = np.array(pil_img.resize(target_size, Image.BILINEAR))
    
    # Convert to 3-channel if needed (for normal backbone)
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = np.array([img, img, img]).transpose(1, 2, 0) if len(img.shape) == 2 else np.repeat(img, 3, axis=2)
    
    transforms =  A.Compose(
        [
            neglog_lin_interp(),
            A.MedianBlur(blur_limit=5),
            mixture_window(keep_original=True, model="kmeans"),
            # clahe(p=0.3),
        ],
        # **kwargs,
    )
        
    # Apply the transformation and record replay data
    transforms = transforms(image=img)
    augmented_img = transforms["image"]

    # results["img"] = augmented_img
    return augmented_img
    

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