###################################################################################################
#
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
SimpleNet_v1 network for AI85/AI86. 
Simplified version of the network proposed in [1].

[1] HasanPour, Seyyed Hossein, et al. "Lets keep it simple, using simple architectures to outperform 
    deeper and more complex architectures." arXiv preprint arXiv:1608.06037 (2016).
"""
import torch.nn as nn
import ai8x


class AI85SimpleNet(nn.Module):
    """
    SimpleNet v1 Model
    """
    def __init__(
            self,
            num_classes=100,
            num_channels=3,
            dimensions=(32, 32),  # pylint: disable=unused-argument
            bias=False,
    ):
        super(AI85SimpleNet, self).__init__()

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 16, 3, stride=1, padding=1, bias=bias)
        self.conv2 = ai8x.FusedConv2dReLU(16, 20, 3, stride=1, padding=1, bias=bias)
        self.conv3 = ai8x.FusedConv2dReLU(20, 20, 3, stride=1, padding=1, bias=bias)
        self.conv4 = ai8x.FusedConv2dReLU(20, 20, 3, stride=1, padding=1, bias=bias)
        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(20, 20, 3, pool_size=2, pool_stride=2, 
                                                 stride=1, padding=1, bias=bias)
        self.conv6 = ai8x.FusedConv2dReLU(20, 20, 3, stride=1, padding=1, bias=bias)
        self.conv7 = ai8x.FusedConv2dReLU(20, 44, 3, stride=1, padding=1, bias=bias)
        self.conv8 = ai8x.FusedMaxPoolConv2dReLU(44, 48, 3, pool_size=2, pool_stride=2, 
                                                 stride=1, padding=1, bias=bias)
        self.conv9 = ai8x.FusedConv2dReLU(48, 48, 3, stride=1, padding=1, bias=bias)
        self.conv10 = ai8x.FusedMaxPoolConv2dReLU(48, 96, 3, pool_size=2, pool_stride=2, 
                                                  stride=1, padding=1, bias=bias)
        self.conv11 = ai8x.FusedMaxPoolConv2dReLU(96, 512, 1, pool_size=2, pool_stride=2, padding=0, bias=bias)
        self.conv12 = ai8x.FusedConv2dReLU(512, 128, 1, stride=1, padding=0, bias=bias)
        self.conv13 = ai8x.FusedMaxPoolConv2dReLU(128, 128, 3, pool_size=2, pool_stride=2, 
                                                  stride=1, padding=1, bias=bias)
        self.conv14 = ai8x.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=bias)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = x.view(x.size(0), -1)
        return x


def ai85simplenet(pretrained=False, **kwargs):
    """
    Constructs a SimpleNet v1 model.
    """
    assert not pretrained
    return AI85SimpleNet(**kwargs)


models = [
    {
        'name': 'ai85simplenet',
        'min_input': 1,
        'dim': 2,
    },
]
