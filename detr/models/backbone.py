# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    efficientnet_b3,
    EfficientNet_B3_Weights,
    EfficientNet_B5_Weights,
    resnet18,
    ResNet18_Weights,
    resnet34,
    ResNet34_Weights,
    resnet50,
    ResNet50_Weights,
)
from .efficientnet import (
    film_efficientnet_b0,
    film_efficientnet_b3,
    film_efficientnet_b5,
)

# detr/models/backbone.py (add near other imports)
from .backbone_xrv import XRV_DenseNet121_Backbone

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias



class BackboneBase(nn.Module):
    def __init__(
        self,
        name,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ):
        super().__init__()
        # The IntermediateLayerGetter logic below removes the last few layers in the CNNs (e.g., average pooling and classification layer)
        # and returns a dictionary-style model. For example, for the EfficientNet, it returns an ordered dictionary where the key is '0'
        # and the value is the model portion preceding the final average pooling and classification layers.
        if "resnet" in name:
            if return_interm_layers:
                return_layers = {
                    "layer1": "0",
                    "layer2": "1",
                    "layer3": "2",
                    "layer4": "3",
                }
            else:
                return_layers = {"layer4": "0"}
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        else:  # efficientnet
            return_layers = {"features": "0"}
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels



class Backbone(BackboneBase):
    """Image encoder backbone."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        # Load pretrained weights.
        if name == "resnet18":
            weights = ResNet18_Weights.DEFAULT
            num_channels = 512
        elif name == "resnet34":
            weights = ResNet34_Weights.DEFAULT
            num_channels = 512
        elif name == "resnet50":
            weights = ResNet50_Weights.DEFAULT
            num_channels = 2048
        elif name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT
            num_channels = 1280
        elif name == "efficientnet_b3":
            weights = EfficientNet_B3_Weights.DEFAULT
            num_channels = 1536
        else:
            raise ValueError
        # Initialize pretrained model.
        if "resnet" in name:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                weights=weights,
                norm_layer=FrozenBatchNorm2d,
            )  # pretrained
        else:  # efficientnet
            backbone = getattr(torchvision.models, name)(
                weights=weights, norm_layer=FrozenBatchNorm2d
            )  # pretrained
        super().__init__(
            name, backbone, train_backbone, num_channels, return_interm_layers
        )
        # Get image preprocessing function.
        self.preprocess = (
            weights.transforms()
        )  # Use this to preprocess images the same way as the pretrained model (e.g., ResNet-18).

    def forward(self, tensor):
        tensor = self.preprocess(tensor)
        xs = self.body(tensor)
        return xs


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos
    
# class Backbone(BackboneBase):
#     """ResNet backbone with frozen BatchNorm."""
#     def __init__(self, name: str,
#                  train_backbone: bool,
#                  return_interm_layers: bool,
#                  dilation: bool,
#                  input_channels: int = 3):
#         # backbone = getattr(torchvision.models, name)(
#         #     replace_stride_with_dilation=[False, False, dilation],
#         #     pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        
#         # backbone = getattr(torchvision.models, name)(
#         #     replace_stride_with_dilation=[False, False, dilation],
#         #     pretrained=True, norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
#         # backbone = getattr(torchvision.models, name)(
#         #     replace_stride_with_dilation=[False, False, dilation],
#         #     weights=ResNet18_Weights.IMAGENET1K_V1, norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
#         backbone = getattr(torchvision.models, name)(
#             replace_stride_with_dilation=[False, False, dilation],
#             weights=ResNet18_Weights.DEFAULT, norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        
#         # Modify first conv layer to accept additional input channels
#         if input_channels != 3:
#             old_conv = backbone.conv1
#             backbone.conv1 = nn.Conv2d(
#                 input_channels, 
#                 old_conv.out_channels,
#                 kernel_size=old_conv.kernel_size,
#                 stride=old_conv.stride,
#                 padding=old_conv.padding,
#                 bias=old_conv.bias is not None
#             )
            
#             # Initialize new conv layer weights
#             with torch.no_grad():
#                 if input_channels > 3:
#                     # Copy pretrained weights for first 3 channels
#                     backbone.conv1.weight[:, :3, :, :] = old_conv.weight
#                     # Initialize additional channels with average of RGB channels
#                     avg_weight = old_conv.weight.mean(dim=1, keepdim=True)
#                     for i in range(3, input_channels):
#                         backbone.conv1.weight[:, i:i+1, :, :] = avg_weight
#                 else:
#                     # If fewer channels, just copy the first few
#                     backbone.conv1.weight = old_conv.weight[:, :input_channels, :, :]
                
#                 if old_conv.bias is not None:
#                     backbone.conv1.bias = old_conv.bias
        
#         num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
#         super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


# def build_backbone(args):
#     position_embedding = build_position_encoding(args)
#     train_backbone = args.lr_backbone > 0
#     return_interm_layers = args.masks
#     backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, input_channels)
#     model = Joiner(backbone, position_embedding)
#     model.num_channels = backbone.num_channels
#     return model

def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks

    if getattr(args, "backbone", "") == "xrv_densenet121":
        backbone = XRV_DenseNet121_Backbone(
            return_interm_layers=return_interm_layers,
            train_backbone=train_backbone,
            use_frozen_bn=False,  # set True if you want FrozenBN behavior
        )
        model = Joiner(backbone, position_embedding)
        model.num_channels = backbone.num_channels
        return model
    else:
        input_channels = getattr(args, 'input_channels', 3)  # Default to 3 if not specified
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, input_channels)
        model = Joiner(backbone, position_embedding)
        model.num_channels = backbone.num_channels
        return model

