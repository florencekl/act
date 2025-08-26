# backbone_xrv.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv

from torchvision.models._utils import IntermediateLayerGetter

# Optional: mirror your FrozenBatchNorm2d if needed by your pipeline
class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class XRV_DenseNet121_Backbone(nn.Module):
    """
    TorchXRayVision DenseNet121 (pretrained on chest X-rays) as DETR backbone.
    Returns dict(str->Tensor) like torchvision IntermediateLayerGetter.
    """
    def __init__(self, return_interm_layers: bool = False, train_backbone: bool = True, use_frozen_bn: bool = False):
        super().__init__()
        # Common strong checkpoint across NIH/CheXpert/MIMIC @ 224x224
        # (xrv wraps torchvision densenet121 under .features)densenet121-res224-chex
        self.net = xrv.models.DenseNet(weights="densenet121-res224-all")
        # self.net = xrv.models.DenseNet(weights="densenet121-res224-chex")
        if not train_backbone:
            for p in self.net.parameters():
                p.requires_grad_(False)

        # TorchXRayVision uses grayscale (1ch). Keep a tiny preproc to accept 1ch or 3ch inputs.
        self.to_gray = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.to_gray.weight[:] = 1/3  # average RGB -> 1ch
        self.accepts_rgb = True  # if you push RGB X-rays, we’ll compress to 1ch

        # Grab the torchvision densenet feature trunk
        feats = self.net.features  # nn.Sequential
        # We’ll expose intermediate stages similar to ResNet layer1..4
        # DenseNet-121 stages approx:
        # 0:conv0,1:norm0,2:relu0,3:pool0,
        # 4:denseblock1,5:transition1,
        # 6:denseblock2,7:transition2,
        # 8:denseblock3,9:transition3,
        # 10:denseblock4,11:norm5
        self.stem = nn.Sequential(*feats[:4])     # 1/2 resolution
        self.stage1 = nn.Sequential(*feats[4:6])  # after denseblock1
        self.stage2 = nn.Sequential(*feats[6:8])  # after denseblock2
        self.stage3 = nn.Sequential(*feats[8:10]) # after denseblock3
        self.stage4 = nn.Sequential(feats[10], feats[11])  # denseblock4 + norm5

        # Optionally swap BN to FrozenBN to match your codebase behavior
        if use_frozen_bn:
            self._freeze_bn(self)

        self.return_interm_layers = return_interm_layers
        # Channels after each stage (DenseNet-121 specifics)
        self.out_channels = {
            "layer1": 256,   # after denseblock1 (w/ transition)
            "layer2": 512,   # after denseblock2
            "layer3": 1024,  # after denseblock3
            "layer4": 1024,  # after norm5 (pre-classifier)
        }
        self.num_channels = self.out_channels["layer4"]

    def _freeze_bn(self, module):
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                fb = FrozenBatchNorm2d(m.num_features)
                fb.weight.data.copy_(m.weight.data)
                fb.bias.data.copy_(m.bias.data)
                fb.running_mean.data.copy_(m.running_mean.data)
                fb.running_var.data.copy_(m.running_var.data)
                # replace in parent
                for name, child in list(m.named_children()):
                    pass
        # Simple approach: set BN eval & no grad
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

    def _ensure_1ch(self, x):
        if x.shape[1] == 1:
            return x
        elif x.shape[1] == 3 and self.accepts_rgb:
            return self.to_gray(x)
        else:
            raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]}")

    def forward(self, x):
        # Expect BCHW, float in [0,1] or [0,255] (xrv is robust; you can normalize upstream)
        x = self._ensure_1ch(x)

        y = self.stem(x)      # /2
        l1 = self.stage1(y)   # /4
        l2 = self.stage2(l1)  # /8
        l3 = self.stage3(l2)  # /16
        l4 = self.stage4(l3)  # /16 (DenseNet keeps stride 32 modest)

        if self.return_interm_layers:
            return {"layer1": l1, "layer2": l2, "layer3": l3, "layer4": l4}
        else:
            return {"layer4": l4}
