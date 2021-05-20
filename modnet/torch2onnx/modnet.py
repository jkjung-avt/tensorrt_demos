"""modnet.py

This is a modified version of:
https://github.com/ZHKKKe/MODNet/blob/master/onnx/modnet_onnx.py

* "scale_factor" replaced by "size" in all F.interpolate()
* SEBlock takes only 1 "channels" argument
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import MobileNetV2Backbone


SUPPORTED_BACKBONES = {'mobilenetv2': MobileNetV2Backbone}


#------------------------------------------------------------------------------
# MODNet Basic Modules
#------------------------------------------------------------------------------

class IBNorm(nn.Module):
    """Combine Instance Norm and Batch Norm into One Layer"""

    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        assert in_channels % 2 == 0
        self.bnorm_channels = in_channels // 2
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)

    def forward(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(x[:, self.bnorm_channels:, ...].contiguous())

        return torch.cat((bn_x, in_x), 1)


class Conv2dIBNormRelu(nn.Module):
    """Convolution + IBNorm + ReLu"""

    def __init__(self, in_channels, out_channels, kernel_size,
                stride=1, padding=0, dilation=1, groups=1, bias=True,
                with_ibn=True, with_relu=True):
        super(Conv2dIBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                    stride=stride, padding=padding, dilation=dilation,
                    groups=groups, bias=bias)
        ]

        if with_ibn:
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SEBlock(nn.Module):
    """SE Block as proposed in https://arxiv.org/pdf/1709.01507.pdf"""

    def __init__(self, channels, reduction=1):
        super(SEBlock, self).__init__()
        self.channels = channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b = x.size()[0]
        w = self.pool(x).view(b, self.channels)
        w = self.fc(w).view(b, self.channels, 1, 1)
        return x * w


#------------------------------------------------------------------------------
# MODNet Branches
#------------------------------------------------------------------------------

class LRBranch(nn.Module):
    """Low Resolution Branch of MODNet"""

    def __init__(self, backbone):
        super(LRBranch, self).__init__()

        enc_channels = backbone.enc_channels

        self.backbone = backbone
        self.se_block = SEBlock(enc_channels[4], reduction=4)
        self.conv_lr16x = Conv2dIBNormRelu(enc_channels[4], enc_channels[3], 5, stride=1, padding=2)
        self.conv_lr8x = Conv2dIBNormRelu(enc_channels[3], enc_channels[2], 5, stride=1, padding=2)
        self.conv_lr = Conv2dIBNormRelu(enc_channels[2], 1, kernel_size=3, stride=2, padding=1, with_ibn=False, with_relu=False)

    def forward(self, img):
        enc_features = self.backbone.forward(img)
        enc2x, enc4x, enc32x = enc_features[0], enc_features[1], enc_features[4]

        enc32x = self.se_block(enc32x)
        h, w = enc32x.size()[2:]  # replacing "scale_factor"
        lr16x = F.interpolate(enc32x, size=(h*2, w*2), mode='bilinear', align_corners=False)
        lr16x = self.conv_lr16x(lr16x)
        h, w = lr16x.size()[2:]   # replacing "scale_factor"
        lr8x = F.interpolate(lr16x, size=(h*2, w*2), mode='bilinear', align_corners=False)
        lr8x = self.conv_lr8x(lr8x)

        return lr8x, [enc2x, enc4x]


class HRBranch(nn.Module):
    """High Resolution Branch of MODNet"""

    def __init__(self, hr_channels, enc_channels):
        super(HRBranch, self).__init__()

        self.tohr_enc2x = Conv2dIBNormRelu(enc_channels[0], hr_channels, 1, stride=1, padding=0)
        self.conv_enc2x = Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=2, padding=1)

        self.tohr_enc4x = Conv2dIBNormRelu(enc_channels[1], hr_channels, 1, stride=1, padding=0)
        self.conv_enc4x = Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1)

        self.conv_hr4x = nn.Sequential(
            Conv2dIBNormRelu(3 * hr_channels + 3, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_hr2x = nn.Sequential(
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_hr = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    def forward(self, img, enc2x, enc4x, lr8x):
        h, w = img.size()[2:]  # replacing "scale_factor"
        assert h % 4 == 0 and w % 4 == 0
        img2x = F.interpolate(img, size=(h//2, w//2), mode='bilinear', align_corners=False)
        img4x = F.interpolate(img, size=(h//4, w//4), mode='bilinear', align_corners=False)

        enc2x = self.tohr_enc2x(enc2x)
        hr4x = self.conv_enc2x(torch.cat((img2x, enc2x), dim=1))

        enc4x = self.tohr_enc4x(enc4x)
        hr4x = self.conv_enc4x(torch.cat((hr4x, enc4x), dim=1))

        h, w = lr8x.size()[2:]  # replacing "scale_factor"
        lr4x = F.interpolate(lr8x, size=(h*2, w*2), mode='bilinear', align_corners=False)
        hr4x = self.conv_hr4x(torch.cat((hr4x, lr4x, img4x), dim=1))

        h, w = hr4x.size()[2:]  # replacing "scale_factor"
        hr2x = F.interpolate(hr4x, size=(h*2, w*2), mode='bilinear', align_corners=False)
        hr2x = self.conv_hr2x(torch.cat((hr2x, enc2x), dim=1))

        return hr2x


class FusionBranch(nn.Module):
    """Fusion Branch of MODNet"""

    def __init__(self, hr_channels, enc_channels):
        super(FusionBranch, self).__init__()
        self.conv_lr4x = Conv2dIBNormRelu(enc_channels[2], hr_channels, 5, stride=1, padding=2)

        self.conv_f2x = Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1)
        self.conv_f = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, int(hr_channels / 2), 3, stride=1, padding=1),
            Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    def forward(self, img, lr8x, hr2x):
        h, w = lr8x.size()[2:]  # replacing "scale_factor"
        lr4x = F.interpolate(lr8x, size=(h*2, w*2), mode='bilinear', align_corners=False)
        lr4x = self.conv_lr4x(lr4x)
        h, w = lr4x.size()[2:]  # replacing "scale_factor"
        lr2x = F.interpolate(lr4x, size=(h*2, w*2), mode='bilinear', align_corners=False)

        f2x = self.conv_f2x(torch.cat((lr2x, hr2x), dim=1))
        h, w = f2x.size()[2:]   # replacing "scale_factor"
        f = F.interpolate(f2x, size=(h*2, w*2), mode='bilinear', align_corners=False)
        f = self.conv_f(torch.cat((f, img), dim=1))
        pred_matte = torch.sigmoid(f)

        return pred_matte


#------------------------------------------------------------------------------
# MODNet
#------------------------------------------------------------------------------

class MODNet(nn.Module):
    """Architecture of MODNet"""

    def __init__(self, in_channels=3, hr_channels=32, backbone_arch='mobilenetv2', backbone_pretrained=False):
        super(MODNet, self).__init__()

        self.in_channels = in_channels
        self.hr_channels = hr_channels
        self.backbone_arch = backbone_arch

        self.backbone = SUPPORTED_BACKBONES[self.backbone_arch](self.in_channels)
        self.lr_branch = LRBranch(self.backbone)
        self.hr_branch = HRBranch(self.hr_channels, self.backbone.enc_channels)
        self.f_branch = FusionBranch(self.hr_channels, self.backbone.enc_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                self._init_norm(m)

        if backbone_pretrained:
            self.backbone.load_pretrained_ckpt()

    def forward(self, img):
        lr8x, [enc2x, enc4x] = self.lr_branch(img)
        hr2x = self.hr_branch(img, enc2x, enc4x, lr8x)
        pred_matte = self.f_branch(img, lr8x, hr2x)
        return pred_matte

    def _init_conv(self, conv):
        nn.init.kaiming_uniform_(
            conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def _init_norm(self, norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)
