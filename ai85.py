###################################################################################################
#
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Contains the limits of the AI85 implementation and custom PyTorch modules that take
the limits into account.
"""

import torch
import torch.nn as nn
import ai84

WEIGHT_BITS = 8
DATA_BITS = 8
ACTIVATION_BITS = 8
FC_ACTIVATION_BITS = 16

WEIGHT_INPUTS = 256
WEIGHT_DEPTH = 768

MAX_AVG_POOL = 16


class Fire(nn.Module):
    """
    AI85 - Fire Layer
    """
    def __init__(self, in_planes, squeeze_planes, expand1x1_planes, expand3x3_planes,
                 bias=True, simulate=False):
        super(Fire, self).__init__()
        self.squeeze_layer = ai84.FusedConv2dReLU(in_channels=in_planes,
                                                  out_channels=squeeze_planes, kernel_size=1,
                                                  bias=bias, simulate=simulate)
        self.expand1x1_layer = ai84.FusedConv2dReLU(in_channels=squeeze_planes,
                                                    out_channels=expand1x1_planes, kernel_size=1,
                                                    bias=bias, simulate=simulate)
        self.expand3x3_layer = ai84.FusedConv2dReLU(in_channels=squeeze_planes,
                                                    out_channels=expand3x3_planes, kernel_size=3,
                                                    padding=1, bias=bias, simulate=simulate)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.squeeze_layer(x)
        return torch.cat([
            self.expand1x1_layer(x),
            self.expand3x3_layer(x)
        ], 1)
