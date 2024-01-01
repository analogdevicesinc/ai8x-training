###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################

import torch.nn as nn

import ai8x


class AI85NASNET_KWS20_RES_1(nn.Module):

    def __init__(
            self,
            num_classes,
            num_channels=128,
            dimensions=(128, 1),
            bias=True,
            **kwargs
    ):
        super().__init__()
        self.conv1_1 = ai8x.FusedConv1dBNReLU(num_channels, 128, 1, stride=1, padding=0, bias=bias, batchnorm="NoAffine", **kwargs)
        self.conv1_2 = ai8x.FusedConv1dBNReLU(128, 64, 3, stride=1, padding=1, bias=bias, batchnorm="NoAffine", **kwargs)
        self.conv1_3 = ai8x.FusedConv1dBNReLU(64, 128, 3, stride=1, padding=1, bias=bias, batchnorm="NoAffine", **kwargs)
        self.conv2_1 = ai8x.FusedMaxPoolConv1dBNReLU(128, 128, 3, stride=1, padding=1, bias=bias, batchnorm="NoAffine", **kwargs)
        self.conv2_2 = ai8x.FusedConv1dBNReLU(128, 64, 1, stride=1, padding=0, bias=bias, batchnorm="NoAffine", **kwargs)
        self.conv2_3 = ai8x.FusedConv1dBNReLU(64, 128, 1, stride=1, padding=0, bias=bias, batchnorm="NoAffine", **kwargs)
        self.conv3_1 = ai8x.FusedMaxPoolConv1dBNReLU(128, 128, 3, stride=1, padding=1, bias=bias, batchnorm="NoAffine", **kwargs)
        self.conv3_2 = ai8x.FusedConv1dBNReLU(128, 64, 5, stride=1, padding=2, bias=bias, batchnorm="NoAffine", **kwargs)
        self.conv4_1 = ai8x.FusedMaxPoolConv1dBNReLU(64, 128, 5, stride=1, padding=2, bias=bias, batchnorm="NoAffine", **kwargs)
        self.conv4_2 = ai8x.FusedConv1dBNReLU(128, 128, 1, stride=1, padding=0, bias=bias, batchnorm="NoAffine", **kwargs)
        self.conv5_1 = ai8x.FusedMaxPoolConv1dBNReLU(128, 128, 5, stride=1, padding=2, bias=bias, batchnorm="NoAffine", **kwargs)
        self.conv5_2 = ai8x.FusedConv1dBNReLU(128, 64, 3, stride=1, padding=1, bias=bias, batchnorm="NoAffine", **kwargs)
        self.conv6_1 = ai8x.FusedMaxPoolConv1dBNReLU(64, 64, 5, stride=1, padding=2, bias=bias, batchnorm="NoAffine", **kwargs)
        self.conv6_2 = ai8x.FusedConv1dBNReLU(64, 128, 1, stride=1, padding=0, bias=bias, batchnorm="NoAffine", **kwargs)
        self.fc = ai8x.Linear(512, num_classes, bias=bias, wide=True, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv6_1(x)
        x = self.conv6_2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ai85nasnet_kws20_res_1(pretrained=False, **kwargs):
    assert not pretrained
    return AI85NASNET_KWS20_RES_1(**kwargs)


models = [
    {
        'name': 'ai85nasnet_kws20_res_1',
        'min_input': 1,
        'dim': 1,
    },
]

