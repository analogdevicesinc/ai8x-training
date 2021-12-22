###################################################################################################
#
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
SimpleNet_v1 network with added residual layers for AI85.
Simplified version of the network proposed in [1].

[1] HasanPour, Seyyed Hossein, et al. "Lets keep it simple, using simple architectures to
    outperform deeper and more complex architectures." arXiv preprint arXiv:1608.06037 (2016).
"""
from torch import nn

import ai8x


class AI85ResidualSimpleNet(nn.Module):
    """
    Residual SimpleNet v1 Model
    """
    def __init__(
            self,
            num_classes=100,
            num_channels=3,
            dimensions=(32, 32),  # pylint: disable=unused-argument
            bias=False,
            **kwargs
    ):
        super().__init__()

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 16, 3, stride=1, padding=1, bias=bias,
                                          **kwargs)
        self.conv2 = ai8x.FusedConv2dReLU(16, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv3 = ai8x.FusedConv2dReLU(20, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv4 = ai8x.FusedConv2dReLU(20, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.resid1 = ai8x.Add()
        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(20, 20, 3, pool_size=2, pool_stride=2,
                                                 stride=1, padding=1, bias=bias, **kwargs)
        self.conv6 = ai8x.FusedConv2dReLU(20, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.resid2 = ai8x.Add()
        self.conv7 = ai8x.FusedConv2dReLU(20, 44, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv8 = ai8x.FusedMaxPoolConv2dReLU(44, 48, 3, pool_size=2, pool_stride=2,
                                                 stride=1, padding=1, bias=bias, **kwargs)
        self.conv9 = ai8x.FusedConv2dReLU(48, 48, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.resid3 = ai8x.Add()
        self.conv10 = ai8x.FusedMaxPoolConv2dReLU(48, 96, 3, pool_size=2, pool_stride=2,
                                                  stride=1, padding=1, bias=bias, **kwargs)
        self.conv11 = ai8x.FusedMaxPoolConv2dReLU(96, 512, 1, pool_size=2, pool_stride=2,
                                                  padding=0, bias=bias, **kwargs)
        self.conv12 = ai8x.FusedConv2dReLU(512, 128, 1, stride=1, padding=0, bias=bias, **kwargs)
        self.conv13 = ai8x.FusedMaxPoolConv2dReLU(128, 128, 3, pool_size=2, pool_stride=2,
                                                  stride=1, padding=1, bias=bias, **kwargs)
        self.conv14 = ai8x.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=bias,
                                  wide=True, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)          # 16x32x32
        x_res = self.conv2(x)      # 20x32x32
        x = self.conv3(x_res)      # 20x32x32
        x = self.resid1(x, x_res)  # 20x32x32
        x = self.conv4(x)          # 20x32x32
        x_res = self.conv5(x)      # 20x16x16
        x = self.conv6(x_res)      # 20x16x16
        x = self.resid2(x, x_res)  # 20x16x16
        x = self.conv7(x)          # 44x16x16
        x_res = self.conv8(x)      # 48x8x8
        x = self.conv9(x_res)      # 48x8x8
        x = self.resid3(x, x_res)  # 48x8x8
        x = self.conv10(x)         # 96x4x4
        x = self.conv11(x)         # 512x2x2
        x = self.conv12(x)         # 128x2x2
        x = self.conv13(x)         # 128x1x1
        x = self.conv14(x)         # num_classesx1x1
        x = x.view(x.size(0), -1)
        return x


def ai85ressimplenet(pretrained=False, **kwargs):
    """
    Constructs a Residual SimpleNet v1 model.
    """
    assert not pretrained
    return AI85ResidualSimpleNet(**kwargs)


models = [
    {
        'name': 'ai85ressimplenet',
        'min_input': 1,
        'dim': 2,
    },
]
