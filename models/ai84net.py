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
from torch import nn

import ai8x


class AI84Net5(nn.Module):
    """
    5-Layer CNN that uses max parameters in AI84
    """
    def __init__(self, num_classes=10, num_channels=3, dimensions=(28, 28),
                 planes=60, pool=2, fc_inputs=12, bias=False):
        super().__init__()

        # AI84 Limits
        assert planes + num_channels <= ai8x.dev.WEIGHT_INPUTS
        assert planes + fc_inputs <= ai8x.dev.WEIGHT_DEPTH-1
        assert pool == 2
        assert dimensions[0] == dimensions[1]  # Only square supported

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

        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(planes, ai8x.dev.WEIGHT_DEPTH-planes-fc_inputs, 3,
                                                 pool_size=2, pool_stride=2, padding=1,
                                                 bias=bias)
        dim //= 2  # pooling, padding 0 -> 8x8
        # padding 1 -> no change in dimensions

        self.conv4 = ai8x.FusedAvgPoolConv2dReLU(ai8x.dev.WEIGHT_DEPTH-planes-fc_inputs,
                                                 fc_inputs, 3,
                                                 pool_size=pool, pool_stride=2, padding=1,
                                                 bias=bias)
        dim //= pool  # pooling, padding 0 -> 4x4
        # padding 1 -> no change in dimensions

        self.fc = ai8x.SoftwareLinear(fc_inputs*dim*dim, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ai84net5(pretrained=False, **kwargs):
    """
    Constructs a AI84Net5 model.
    """
    assert not pretrained
    return AI84Net5(**kwargs)


class AI84NetExtraSmall(nn.Module):
    """
    Minimal CNN that tries to achieve 1uJ per inference for MNIST
    """
    def __init__(self, num_classes=10, num_channels=3, dimensions=(28, 28),
                 fc_inputs=8, bias=False):
        super().__init__()

        # AI84 Limits
        assert dimensions[0] == dimensions[1]  # Only square supported

        # Keep track of image dimensions so one constructor works for all image sizes
        dim = dimensions[0]

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 8, 3,
                                          padding=1, bias=bias)
        # padding 1 -> no change in dimensions -> 8x28x28

        pad = 2 if dim == 28 else 1
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(8, 8, 3, pool_size=2, pool_stride=2,
                                                 padding=pad, bias=bias)
        dim //= 2  # pooling, padding 0 -> 8x14x14
        if pad == 2:
            dim += 2  # padding 2 -> 8x16x16

        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(8, fc_inputs, 3,
                                                 pool_size=4, pool_stride=4, padding=1,
                                                 bias=bias)
        dim //= 4  # pooling, padding 0 -> 8x4x4
        # padding 1 -> 8x4x4

        self.fc = ai8x.SoftwareLinear(fc_inputs*dim*dim, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ai84netextrasmall(pretrained=False, **kwargs):
    """
    Constructs a AI84NetExtraSmall model.
    """
    assert not pretrained
    return AI84NetExtraSmall(**kwargs)


class AI84NetSmall(nn.Module):
    """
    Minimal CNN that tries to achieve 1uJ per inference for MNIST
    """
    def __init__(self, num_classes=10, num_channels=3, dimensions=(28, 28),
                 fc_inputs=12, bias=False):
        super().__init__()

        # AI84 Limits
        assert dimensions[0] == dimensions[1]  # Only square supported

        # Keep track of image dimensions so one constructor works for all image sizes
        dim = dimensions[0]

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 16, 3,
                                          padding=1, bias=bias)
        # padding 1 -> no change in dimensions -> 16x28x28

        pad = 2 if dim == 28 else 1
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(16, 16, 3, pool_size=2, pool_stride=2,
                                                 padding=pad, bias=bias)
        dim //= 2  # pooling, padding 0 -> 16x14x14
        if pad == 2:
            dim += 2  # padding 2 -> 16x16x16

        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(16, fc_inputs, 3,
                                                 pool_size=4, pool_stride=4, padding=1,
                                                 bias=bias)
        dim //= 4  # pooling, padding 0 -> 16x4x4
        # padding 1 -> 12x4x4

        self.fc = ai8x.SoftwareLinear(fc_inputs*dim*dim, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ai84netsmall(pretrained=False, **kwargs):
    """
    Constructs a AI84NetSmall model.
    """
    assert not pretrained
    return AI84NetSmall(**kwargs)


class AI84Net7(nn.Module):
    """
    CNN that tries to achieve accuracy > %90 for kws.
    """
    def __init__(self, num_classes=7, num_channels=1, dimensions=(64, 64),
                 fc_inputs=30, bias=False):
        super().__init__()

        # AI84 Limits
        assert dimensions[0] == dimensions[1]  # Only square supported

        # Keep track of image dimensions so one constructor works for all image sizes
        dim = dimensions[0]

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 15, 3,
                                          padding=1, bias=bias)
        # padding 1 -> no change in dimensions -> 15x28x28

        pad = 2 if dim == 28 else 1
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(15, 30, 3, pool_size=2, pool_stride=2,
                                                 padding=pad, bias=bias)
        dim //= 2  # pooling, padding 0 -> 30x14x14
        if pad == 2:
            dim += 2  # padding 2 -> 30x16x16

        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(30, 60, 3, pool_size=2, pool_stride=2, padding=1,
                                                 bias=bias)
        dim //= 2  # pooling, padding 0 -> 60x8x8

        self.conv4 = ai8x.FusedMaxPoolConv2dReLU(60, 30, 3, pool_size=2, pool_stride=2, padding=1,
                                                 bias=bias)
        dim //= 2  # pooling, padding 0 -> 30x4x4

        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(30, 30, 3, pool_size=2, pool_stride=2, padding=1,
                                                 bias=bias)
        dim //= 2  # pooling, padding 0 -> 30x2x2

        self.conv6 = ai8x.FusedConv2dReLU(30, fc_inputs, 3, padding=1, bias=bias)

        self.fc = ai8x.SoftwareLinear(fc_inputs*dim*dim, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ai84net7(pretrained=False, **kwargs):
    """
    Constructs a AI84Net7 model.
    """
    assert not pretrained
    return AI84Net7(**kwargs)


models = [
    {
        'name': 'ai84net5',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai84netsmall',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai84netextrasmall',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai84net7',
        'min_input': 1,
        'dim': 2,
    },
]
