###################################################################################################
#
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
FaceID network for AI85/AI86

Optionally quantize/clamp activations
"""
import torch.nn as nn
import ai84


class AI85FaceIDNet(nn.Module):
    """
    Simple FaceNet Model
    """
    def __init__(
            self,
            num_classes=None,  # pylint: disable=unused-argument
            num_channels=3,
            dimensions=(160, 120),  # pylint: disable=unused-argument
            simulate=False,
            bias=True,
            device=85,
    ):
        super(AI85FaceIDNet, self).__init__()

        self.conv1 = ai84.FusedConv2dReLU(num_channels, 16, 3, padding=1,
                                          bias=bias, simulate=simulate)
        self.conv2 = ai84.FusedMaxPoolConv2dReLU(16, 32, 3, pool_size=2, pool_stride=2,
                                                 padding=1, bias=bias, simulate=simulate,
                                                 device=device)
        self.conv3 = ai84.FusedMaxPoolConv2dReLU(32, 32, 3, pool_size=2, pool_stride=2,
                                                 padding=1, bias=bias, simulate=simulate,
                                                 device=device)
        self.conv4 = ai84.FusedMaxPoolConv2dReLU(32, 64, 3, pool_size=2, pool_stride=2,
                                                 padding=1, bias=bias, simulate=simulate,
                                                 device=device)
        self.conv5 = ai84.FusedMaxPoolConv2dReLU(64, 64, 3, pool_size=2, pool_stride=2,
                                                 padding=1, bias=bias, simulate=simulate,
                                                 device=device)
        self.conv6 = ai84.FusedConv2dReLU(64, 64, 3, padding=1,
                                          bias=bias, simulate=simulate)
        self.conv7 = ai84.FusedConv2dReLU(64, 64, 3, padding=1,
                                          bias=bias, simulate=simulate)
        self.conv8 = ai84.FusedMaxPoolConv2d(64, 512, 1, pool_size=2, pool_stride=2,
                                             padding=0, bias=False, simulate=simulate,
                                             device=device)
        self.avgpool = nn.AvgPool2d((5, 3))

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.avgpool(x)
        return x


def ai85faceidnet(pretrained=False, **kwargs):
    """
    Constructs a FaceIDNet model.
    """
    assert not pretrained
    return AI85FaceIDNet(**kwargs)
