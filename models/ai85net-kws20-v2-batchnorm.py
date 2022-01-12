###################################################################################################
#
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Keyword spotting network for AI85/AI86
"""
from torch import nn

import ai8x


class AI85KWS20Netv2Batchnorm(nn.Module):
    """
    Compound KWS20 v2 Audio net, all with Conv1Ds with batchnorm layers
    """

    # num_classes = n keywords + 1 unknown
    def __init__(
            self,
            num_classes=21,
            num_channels=128,
            dimensions=(128, 1),  # pylint: disable=unused-argument
            bias=False,
            **kwargs
    ):
        super().__init__()
        # T: 128 F :128
        self.conv1 = ai8x.FusedConv1dBNReLU(num_channels, 100, 1, stride=1, padding=0,
                                            bias=bias, batchnorm='Affine', **kwargs)
        # T:  128 F: 100
        self.conv2 = ai8x.FusedConv1dBNReLU(100, 48, 3, stride=1, padding=0,
                                            bias=bias, batchnorm='Affine', **kwargs)
        # T: 126 F : 48
        self.conv3 = ai8x.FusedMaxPoolConv1dBNReLU(48, 96, 3, stride=1, padding=1,
                                                   bias=bias, batchnorm='Affine', **kwargs)
        # T: 62 F : 96
        self.conv4 = ai8x.FusedConv1dBNReLU(96, 128, 3, stride=1, padding=0,
                                            bias=bias, batchnorm='Affine', **kwargs)
        # T : 60 F : 128
        self.conv5 = ai8x.FusedMaxPoolConv1dBNReLU(128, 160, 3, stride=1, padding=1,
                                                   bias=bias, batchnorm='Affine', **kwargs)
        # T: 30 F : 160
        self.conv6 = ai8x.FusedConv1dBNReLU(160, 192, 3, stride=1, padding=0,
                                            bias=bias, batchnorm='Affine', **kwargs)
        # T: 28 F : 192
        self.conv7 = ai8x.FusedAvgPoolConv1dBNReLU(192, 192, 3, stride=1, padding=1,
                                                   bias=bias, batchnorm='Affine', **kwargs)
        # T : 14 F: 256
        self.conv8 = ai8x.FusedConv1dBNReLU(192, 32, 3, stride=1, padding=0,
                                            bias=bias, batchnorm='Affine', **kwargs)
        # T: 12 F : 32
        self.fc = ai8x.Linear(32 * 12, num_classes, bias=bias, wide=True, **kwargs)

        # T: 1 F : 256

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # Run CNN
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ai85kws20netv2batchnorm(pretrained=False, **kwargs):
    """
    Constructs a AI85KWS20Net model.
    rn AI85KWS20Net(**kwargs)
    """
    assert not pretrained
    return AI85KWS20Netv2Batchnorm(**kwargs)


models = [
    {
        'name': 'ai85kws20netv2batchnorm',
        'min_input': 1,
        'dim': 1,
    },
]
