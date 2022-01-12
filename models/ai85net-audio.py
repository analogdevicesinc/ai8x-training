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
from torch import nn

import ai8x


class AI85AudioNet(nn.Module):
    """
    Compound Audio Net, starting with Conv1Ds with kernel_size=1 and then switching to Conv2Ds
    """
    # num_classes = n keywords + 1 unknown
    def __init__(
            self,
            num_classes=21,
            num_channels=128,
            dimensions=(128, 1),  # pylint: disable=unused-argument
            fc_inputs=7,
            bias=False,
            **kwargs
    ):
        super().__init__()

        self.mfcc_conv1 = ai8x.FusedConv1dReLU(num_channels, 100, 1, stride=1, padding=0,
                                               bias=bias, **kwargs)

        self.mfcc_conv2 = ai8x.FusedConv1dReLU(100, 50, 1, stride=1, padding=0,
                                               bias=bias, **kwargs)

        self.mfcc_conv3 = ai8x.FusedConv1dReLU(50, 50, 1, stride=1, padding=0,
                                               bias=bias, **kwargs)

        self.mfcc_conv4 = ai8x.FusedConv1dReLU(50, 50, 1, stride=1, padding=0,
                                               bias=bias, **kwargs)

        self.mfcc_conv5 = ai8x.FusedConv1dReLU(50, 50, 1, stride=1, padding=0,
                                               bias=bias, **kwargs)

        self.mfcc_conv6 = ai8x.FusedConv1dReLU(50, 16, 1, stride=1, padding=0,
                                               bias=bias, **kwargs)

        self.kws_conv1 = ai8x.FusedConv2dReLU(16, 15, 3, stride=1, padding=1,
                                              bias=bias, **kwargs)

        self.kws_conv2 = ai8x.FusedConv2dReLU(15, 30, 3, stride=1, padding=1,
                                              bias=bias, **kwargs)

        self.kws_conv3 = ai8x.FusedConv2dReLU(30, 60, 3, stride=1,
                                              padding=1, bias=bias, **kwargs)

        self.kws_conv4 = ai8x.FusedConv2dReLU(60, 30, 3, stride=1,
                                              padding=1, bias=bias, **kwargs)

        self.kws_conv5 = ai8x.FusedConv2dReLU(30, 30, 3, padding=1, bias=bias, **kwargs)

        self.kws_conv6 = ai8x.FusedConv2dReLU(30, fc_inputs, 3, stride=1,
                                              padding=1, bias=bias, **kwargs)

        self.fc = ai8x.Linear(fc_inputs*128, num_classes, bias=bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # Save input sample for CNN generation
        # import numpy as np
        # a = x.cpu().numpy().squeeze().astype('int64')
        # a = np.clip(a, -128, 127)
        # np.save('sample_speechcomfolded1d_20', a, allow_pickle=False, fix_imports=False)
        # Run CNN
        x = self.mfcc_conv1(x)
        x = self.mfcc_conv2(x)
        x = self.mfcc_conv3(x)
        x = self.mfcc_conv4(x)
        x = self.mfcc_conv5(x)
        x = self.mfcc_conv6(x)
        x = x.view(x.shape[0], x.shape[1], 16, -1)
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


models = [
    {
        'name': 'ai85audionet',
        'min_input': 1,
        'dim': 1,
    },
]
