###################################################################################################
#
# Copyright (C) 2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
UNet network for MAX7800X
"""
import torch
from torch import nn

import ai8x


class AI85UNetSmall(nn.Module):
    """
    Small size UNet model
    """
    def __init__(
            self,
            num_classes=4,
            num_channels=3,
            dimensions=(128, 128),  # pylint: disable=unused-argument
            bias=True,
            **kwargs
    ):
        super().__init__()

        self.enc1 = ai8x.FusedConv2dBNReLU(num_channels, 4, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)
        self.enc2 = ai8x.FusedMaxPoolConv2dBNReLU(4, 8, 3, stride=1, padding=1,
                                                  bias=bias, batchnorm='NoAffine', **kwargs)
        self.enc3 = ai8x.FusedMaxPoolConv2dBNReLU(8, 32, 3, stride=1, padding=1,
                                                  bias=bias, batchnorm='NoAffine', **kwargs)

        self.bneck = ai8x.FusedMaxPoolConv2dBNReLU(32, 64, 3, stride=1, padding=1,
                                                   bias=bias, batchnorm='NoAffine', **kwargs)

        self.upconv3 = ai8x.ConvTranspose2d(64, 32, 3, stride=2, padding=1)
        self.dec3 = ai8x.FusedConv2dBNReLU(64, 32, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        self.upconv2 = ai8x.ConvTranspose2d(32, 8, 3, stride=2, padding=1)
        self.dec2 = ai8x.FusedConv2dBNReLU(16, 8, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        self.upconv1 = ai8x.ConvTranspose2d(8, 4, 3, stride=2, padding=1)
        self.dec1 = ai8x.FusedConv2dBNReLU(8, 16, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv = ai8x.FusedConv2dBN(16, num_classes, 1, stride=1, padding=0,
                                       bias=bias, batchnorm='NoAffine', **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # Run CNN
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        bottleneck = self.bneck(enc3)

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        return self.conv(dec1)


class AI85UNetMedium(nn.Module):
    """
    Medium size UNet model
    """
    def __init__(
            self,
            num_classes=4,
            num_channels=3,
            dimensions=(128, 128),  # pylint: disable=unused-argument
            bias=True,
            **kwargs
    ):
        super().__init__()

        self.enc1 = ai8x.FusedConv2dBNReLU(num_channels, 4, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)
        self.enc2 = ai8x.FusedMaxPoolConv2dBNReLU(4, 8, 3, stride=1, padding=1,
                                                  bias=bias, batchnorm='NoAffine', **kwargs)
        self.enc3 = ai8x.FusedMaxPoolConv2dBNReLU(8, 32, 3, stride=1, padding=1,
                                                  bias=bias, batchnorm='NoAffine', **kwargs)

        self.bneck = ai8x.FusedMaxPoolConv2dBNReLU(32, 64, 3, stride=1, padding=1,
                                                   bias=bias, batchnorm='NoAffine', **kwargs)

        self.upconv3 = ai8x.ConvTranspose2d(64, 32, 3, stride=2, padding=1)
        self.dec3 = ai8x.FusedConv2dBNReLU(64, 60, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        self.upconv2 = ai8x.ConvTranspose2d(60, 8, 3, stride=2, padding=1)
        self.dec2 = ai8x.FusedConv2dBNReLU(16, 48, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        self.upconv1 = ai8x.ConvTranspose2d(48, 4, 3, stride=2, padding=1)
        self.dec1 = ai8x.FusedConv2dBNReLU(8, 64, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        self.dec0 = ai8x.FusedConv2dBNReLU(64, 32, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv = ai8x.FusedConv2dBN(32, num_classes, 1, stride=1, padding=0,
                                       bias=bias, batchnorm='NoAffine', **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # Run CNN
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        bottleneck = self.bneck(enc3)

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        dec0 = self.dec0(dec1)
        return self.conv(dec0)


class AI85UNetLarge(nn.Module):
    """
    Large size UNet model. This model also enables the use of folded data.
    """
    def __init__(
            self,
            num_classes=4,
            num_channels=48,
            dimensions=(88, 88),  # pylint: disable=unused-argument
            bias=True,
            fold_ratio=4,
            **kwargs
    ):
        super().__init__()
        self.fold_ratio = fold_ratio
        self.num_classes = num_classes
        self.num_final_channels = num_classes * fold_ratio * fold_ratio

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, 64, 1, stride=1, padding=0,
                                            bias=bias, batchnorm='NoAffine', **kwargs)
        self.prep1 = ai8x.FusedConv2dBNReLU(64, 64, 1, stride=1, padding=0,
                                            bias=bias, batchnorm='NoAffine', **kwargs)
        self.prep2 = ai8x.FusedConv2dBNReLU(64, 32, 1, stride=1, padding=0,
                                            bias=bias, batchnorm='NoAffine', **kwargs)

        self.enc1 = ai8x.FusedConv2dBNReLU(32, 8, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)
        self.enc2 = ai8x.FusedMaxPoolConv2dBNReLU(8, 28, 3, stride=1, padding=1,
                                                  bias=bias, batchnorm='NoAffine', **kwargs)
        self.enc3 = ai8x.FusedMaxPoolConv2dBNReLU(28, 56, 3, stride=1, padding=1,
                                                  bias=bias, batchnorm='NoAffine', **kwargs)

        self.bneck = ai8x.FusedMaxPoolConv2dBNReLU(56, 112, 3, stride=1, padding=1,
                                                   bias=bias, batchnorm='NoAffine', **kwargs)

        self.upconv3 = ai8x.ConvTranspose2d(112, 56, 3, stride=2, padding=1)
        self.dec3 = ai8x.FusedConv2dBNReLU(112, 56, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        self.upconv2 = ai8x.ConvTranspose2d(56, 28, 3, stride=2, padding=1)
        self.dec2 = ai8x.FusedConv2dBNReLU(56, 28, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        self.upconv1 = ai8x.ConvTranspose2d(28, 8, 3, stride=2, padding=1)
        self.dec1 = ai8x.FusedConv2dBNReLU(16, 48, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        self.dec0 = ai8x.FusedConv2dBNReLU(48, 64, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv_p1 = ai8x.FusedConv2dBNReLU(64, 64, 1, stride=1, padding=0,
                                              bias=bias, batchnorm='NoAffine', **kwargs)
        self.conv_p2 = ai8x.FusedConv2dBNReLU(64, 64, 1, stride=1, padding=0,
                                              bias=bias, batchnorm='NoAffine', **kwargs)
        self.conv_p3 = ai8x.FusedConv2dBN(64, 64, 1, stride=1, padding=0,
                                          bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv = ai8x.FusedConv2dBN(64, self.num_final_channels, 1, stride=1, padding=0,
                                       bias=bias, batchnorm='NoAffine', **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # Run CNN
        x = self.prep0(x)
        x = self.prep1(x)
        x = self.prep2(x)

        enc1 = self.enc1(x)                    # 8x(dim1)x(dim2)
        enc2 = self.enc2(enc1)                 # 28x(dim1/2)x(dim2/2)
        enc3 = self.enc3(enc2)                 # 56x(dim1/4)x(dim2/4)

        bottleneck = self.bneck(enc3)          # 112x(dim1/8)x(dim2/8)

        dec3 = self.upconv3(bottleneck)        # 56x(dim1/4)x(dim2/4)
        dec3 = torch.cat((dec3, enc3), dim=1)  # 112x(dim1/4)x(dim2/4)
        dec3 = self.dec3(dec3)                 # 56x(dim1/4)x(dim2/4)
        dec2 = self.upconv2(dec3)              # 28x(dim1/2)x(dim2/2)
        dec2 = torch.cat((dec2, enc2), dim=1)  # 56(dim1/2)x(dim2/2)
        dec2 = self.dec2(dec2)                 # 28x(dim1/2)x(dim2/2)
        dec1 = self.upconv1(dec2)              # 8x(dim1)x(dim2)
        dec1 = torch.cat((dec1, enc1), dim=1)  # 16x(dim1)x(dim2)
        dec1 = self.dec1(dec1)                 # 48x(dim1)x(dim2)
        dec0 = self.dec0(dec1)                 # 64x(dim1)x(dim2)

        dec0 = self.conv_p1(dec0)
        dec0 = self.conv_p2(dec0)
        dec0 = self.conv_p3(dec0)
        dec0 = self.conv(dec0)                 # num_final_channelsx(dim1)x(dim2)

        return dec0


def ai85unetsmall(pretrained=False, **kwargs):
    """
    Constructs a small unet (unet_v3) model.
    """
    assert not pretrained
    return AI85UNetSmall(**kwargs)


def ai85unetmedium(pretrained=False, **kwargs):
    """
    Constructs a medium unet (unet_v5) model.
    """
    assert not pretrained
    return AI85UNetMedium(**kwargs)


def ai85unetlarge(pretrained=False, **kwargs):
    """
    Constructs a large unet (unet_v7) model.
    """
    assert not pretrained
    return AI85UNetLarge(**kwargs)


models = [
    {
        'name': 'ai85unetsmall',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85unetmedium',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85unetlarge',
        'min_input': 1,
        'dim': 2,
    },
]
