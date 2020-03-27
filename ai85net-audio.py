###################################################################################################
#
# Copyright (C) 2019-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Test networks for AI85/AI86

Optionally quantize/clamp activations
"""
import torch.nn as nn
import ai84


class AI85AudioNet(nn.Module):
    """
    Compound Audio Net, starting with Conv1Ds with kernel_size=1 and then switching to Conv2Ds
    """
    def __init__(self, num_classes=7, num_channels=512, dimensions=(64, 1),
                 simulate=False, fc_inputs=10, bias=False):
        super(AI85AudioNet, self).__init__()

        dim1 = dimensions[0]
        self.mfcc_conv1 = ai84.FusedConv1dReLU(num_channels, 200, 1, stride=1, padding=0,
                                               bias=bias, simulate=simulate, device=85)

        self.mfcc_conv2 = ai84.FusedConv1dReLU(200, 50, 1, stride=1, padding=0,
                                               bias=bias, simulate=simulate, device=85)

        self.mfcc_conv3 = ai84.FusedConv1dReLU(50, 50, 1, stride=1, padding=0,
                                               bias=bias, simulate=simulate, device=85)

        self.mfcc_conv4 = ai84.FusedConv1dReLU(50, 50, 1, stride=1, padding=0,
                                               bias=bias, simulate=simulate, device=85)

        self.mfcc_conv5 = ai84.FusedConv1dReLU(50, 50, 1, stride=1, padding=0,
                                               bias=bias, simulate=simulate, device=85)

        self.mfcc_conv6 = ai84.FusedConv1dReLU(50, 16, 1, stride=1, padding=0,
                                               bias=bias, simulate=simulate, device=85)

        dim2 = 16
        self.kws_conv1 = ai84.FusedConv2dReLU(1, 15, 3, stride=1, padding=1,
                                              bias=bias, simulate=simulate)

        self.kws_conv2 = ai84.FusedConv2dReLU(15, 30, 3, padding=1,
                                              bias=bias, simulate=simulate)

        self.kws_conv3 = ai84.FusedMaxPoolConv2dReLU(30, 60, 3, pool_size=2, pool_stride=2,
                                                     padding=1, bias=bias, simulate=simulate)
        dim1 //= 2
        dim2 //= 2

        self.kws_conv4 = ai84.FusedMaxPoolConv2dReLU(60, 30, 3, pool_size=2, pool_stride=2,
                                                     padding=1, bias=bias, simulate=simulate)
        dim1 //= 2
        dim2 //= 2

        self.kws_conv5 = ai84.FusedConv2dReLU(30, 30, 3, padding=1,
                                              bias=bias, simulate=simulate)

        self.kws_conv6 = ai84.FusedMaxPoolConv2dReLU(30, fc_inputs, 3, pool_size=2, pool_stride=2,
                                                     padding=1, bias=bias, simulate=simulate)
        dim1 //= 2
        dim2 //= 2

        self.fc = ai84.SoftwareLinear(fc_inputs*dim1*dim2, num_classes, bias=bias,
                                      simulate=simulate)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.mfcc_conv1(x)
        x = self.mfcc_conv2(x)
        x = self.mfcc_conv3(x)
        x = self.mfcc_conv4(x)
        x = self.mfcc_conv5(x)
        x = self.mfcc_conv6(x)
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.kws_conv1(x)
        x = self.kws_conv2(x)
        x = self.kws_conv3(x)
        x = self.kws_conv4(x)
        x = self.kws_conv5(x)
        x = self.kws_conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ai85audionet(pretrained=False, **kwargs):
    """
    Constructs a AI85AudioNet model.
    """
    assert not pretrained
    return AI85AudioNet(**kwargs)
