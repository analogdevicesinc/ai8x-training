###################################################################################################
#
# Copyright (C) 2023 Analog Devices, Inc. All Rights Reserved.
# This software is proprietary to Analog Devices, Inc. and its licensors.
#
###################################################################################################
"""
Bayer to Rgb network for AI85
"""
from torch import nn

import ai8x


class bayer2rgbnet(nn.Module):
    """
    Bayer to RGB Network Model
    """
    def __init__(
            self,
            num_classes=None,  # pylint: disable=unused-argument
            num_channels=4,
            dimensions=(64, 64),  # pylint: disable=unused-argument
            bias=False,
            **kwargs):  # pylint: disable=unused-argument

        super().__init__()

        self.l1 = ai8x.Conv2d(num_channels, 3, kernel_size=1, padding=0, bias=bias)
        self.l2 = ai8x.ConvTranspose2d(3, 3, kernel_size=3, padding=1, stride=2, bias=bias)
        self.l3 = ai8x.Conv2d(3, 3, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        """Forward prop"""
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x


def b2rgb(pretrained: bool = False, **kwargs):
    """
    Constructs a bayer2rgbnet model
    """
    assert not pretrained
    return bayer2rgbnet(**kwargs)


models = [
    {
        'name': 'bayer2rgbnet',
        'min_input': 1,
        'dim': 2,
    },
]
