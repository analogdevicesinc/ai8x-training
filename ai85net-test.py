###################################################################################################
#
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
###################################################################################################
"""
Test networks for AI85/AI6

Optionally quantize/clamp activations
"""
import torch.nn as nn
import ai84


class AI85NetWide(nn.Module):
    """
    CNN that uses wide output layer in AI85
    """
    def __init__(self, num_classes=10, num_channels=3, dimensions=(28, 28),
                 simulate=False, planes=128, pool=2, fc_inputs=12, bias=False):
        super(AI85NetWide, self).__init__()

        # Keep track of image dimensions so one constructor works for all image sizes
        dim = dimensions[0]

        self.conv1 = ai84.FusedConv2dReLU(num_channels, planes, 3,
                                          padding=1, bias=bias, simulate=simulate)
        # padding 1 -> no change in dimensions -> MNIST: 28x28 | CIFAR: 32x32

        pad = 2 if dim == 28 else 1
        self.conv2 = ai84.FusedMaxPoolConv2dReLU(planes, 60, 3, pool_size=2, pool_stride=2,
                                                 padding=pad, bias=bias, simulate=simulate)
        dim //= 2  # pooling, padding 0 -> MNIST: 14x14 | CIFAR: 16x16
        if pad == 2:
            dim += 2  # MNIST: padding 2 -> 16x16 | CIFAR: padding 1 -> 16x16

        self.conv3 = ai84.FusedMaxPoolConv2dReLU(60, 56, 3,
                                                 pool_size=2, pool_stride=2, padding=1,
                                                 bias=bias, simulate=simulate)
        dim //= 2  # pooling, padding 0 -> 8x8
        # padding 1 -> no change in dimensions

        self.conv4 = ai84.FusedAvgPoolConv2dReLU(56, fc_inputs, 3,
                                                 pool_size=pool, pool_stride=2, padding=1,
                                                 bias=bias, simulate=simulate)
        dim //= pool  # pooling, padding 0 -> 4x4
        # padding 1 -> no change in dimensions

        self.fc = ai84.SoftwareLinear(fc_inputs*dim*dim, num_classes, bias=True, simulate=simulate)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ai85netwide(pretrained=False, **kwargs):
    """
    Constructs a AI85NetWide model.
    """
    assert not pretrained
    return AI85NetWide(**kwargs)


