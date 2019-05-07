###################################################################################################
#
# Copyright (C) 2019-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Networks that fit into AI84

Optionally quantize/clamp activations
"""
import torch.nn as nn
import ai8x


class AI85Net5(nn.Module):
    """
    5-Layer CNN that uses max parameters in AI84
    """
    def __init__(self, num_classes=10, num_channels=3, dimensions=(28, 28),
                 planes=60, pool=2, fc_inputs=12, bias=False):
        super(AI85Net5, self).__init__()

        # Limits
        assert planes + num_channels <= ai8x.dev.WEIGHT_INPUTS
        assert planes + fc_inputs <= ai8x.dev.WEIGHT_DEPTH-1

        # Keep track of image dimensions so one constructor works for all image sizes
        dim = dimensions[0]

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, planes, 3,
                                          padding=1, bias=bias)
        # padding 1 -> no change in dimensions -> MNIST: 28x28 | CIFAR: 32x32

        pad = 2 if dim == 28 else 1
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(planes, planes, 3, pool_size=2, pool_stride=2,
                                                 padding=pad, bias=bias)
        dim //= 2  # pooling, padding 0 -> MNIST: 14x14 | CIFAR: 16x16
        if pad == 2:
            dim += 2  # MNIST: padding 2 -> 16x16 | CIFAR: padding 1 -> 16x16

        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(planes, 128-planes-fc_inputs, 3,
                                                 pool_size=2, pool_stride=2, padding=1,
                                                 bias=bias)
        dim //= 2  # pooling, padding 0 -> 8x8
        # padding 1 -> no change in dimensions

        self.conv4 = ai8x.FusedAvgPoolConv2dReLU(128-planes-fc_inputs,
                                                 fc_inputs, 3,
                                                 pool_size=pool, pool_stride=2, padding=1,
                                                 bias=bias)
        dim //= pool  # pooling, padding 0 -> 4x4
        # padding 1 -> no change in dimensions

        self.fc = ai8x.Linear(fc_inputs*dim*dim, num_classes, bias=True, wide=True)

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


def ai85net5(pretrained=False, **kwargs):
    """
    Constructs a AI85Net5 model.
    """
    assert not pretrained
    return AI85Net5(**kwargs)


models = [
    {
        'name': 'ai85net5',
        'min_input': 1,
        'dim': 2,
    },
]
