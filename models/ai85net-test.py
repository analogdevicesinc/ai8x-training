###################################################################################################
#
# Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
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
import ai8x_blocks


class AI85NetWide(nn.Module):
    """
    CNN that uses wide output layer in AI85
    """
    def __init__(self, num_classes=10, num_channels=3, dimensions=(28, 28),
                 planes=128, pool=2, fc_inputs=12, bias=False, **kwargs):
        super().__init__()

        # Keep track of image dimensions so one constructor works for all image sizes
        dim = dimensions[0]

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, planes, 3,
                                          padding=1, bias=bias, **kwargs)
        # padding 1 -> no change in dimensions -> MNIST: 28x28 | CIFAR: 32x32

        pad = 2 if dim == 28 else 1
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(planes, 60, 3, pool_size=2, pool_stride=2,
                                                 padding=pad, bias=bias, **kwargs)
        dim //= 2  # pooling, padding 0 -> MNIST: 14x14 | CIFAR: 16x16
        if pad == 2:
            dim += 2  # MNIST: padding 2 -> 16x16 | CIFAR: padding 1 -> 16x16

        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(60, 56, 3,
                                                 pool_size=2, pool_stride=2, padding=1,
                                                 bias=bias, **kwargs)
        dim //= 2  # pooling, padding 0 -> 8x8
        # padding 1 -> no change in dimensions

        self.conv4 = ai8x.FusedAvgPoolConv2dReLU(56, fc_inputs, 3,
                                                 pool_size=pool, pool_stride=2, padding=1,
                                                 bias=bias, **kwargs)
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


def ai85netwide(pretrained=False, **kwargs):
    """
    Constructs a AI85NetWide model with 128 output channels.
    """
    assert not pretrained
    return AI85NetWide(**kwargs)


def ai85net80wide(pretrained=False, **kwargs):
    """
    Constructs a AI85NetWide model with 80 output channels.
    """
    assert not pretrained
    return AI85NetWide(planes=80, **kwargs)


class AI85NetExpansion(nn.Module):
    """
    CNN that uses wide output layer in AI85, and is small enough to fit into data memory with
    32-bit values.
    """
    def __init__(self, num_classes=10, num_channels=3, dimensions=(28, 28),
                 planes=80, pool=2, fc_inputs=12, bias=False, **kwargs):
        super().__init__()

        # Keep track of image dimensions so one constructor works for all image sizes
        dim = dimensions[0]

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 16, 3,
                                          padding=1, bias=bias, **kwargs)
        # padding 1 -> no change in dimensions -> MNIST: 28x28 | CIFAR: 32x32

        pad = 2 if dim == 28 else 1
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(16, planes, 3, pool_size=2, pool_stride=2,
                                                 padding=pad, bias=bias, **kwargs)
        dim //= 2  # pooling, padding 0 -> MNIST: 14x14 | CIFAR: 16x16
        if pad == 2:
            dim += 2  # MNIST: padding 2 -> 16x16 | CIFAR: padding 1 -> 16x16

        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(planes, 16, 3,
                                                 pool_size=2, pool_stride=2, padding=1,
                                                 bias=bias, **kwargs)
        dim //= 2  # pooling, padding 0 -> 8x8
        # padding 1 -> no change in dimensions

        self.conv4 = ai8x.FusedAvgPoolConv2dReLU(16, fc_inputs, 3,
                                                 pool_size=pool, pool_stride=2, padding=1,
                                                 bias=bias, **kwargs)
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


def ai85net80expansion(pretrained=False, **kwargs):
    """
    Constructs a AI85NetExpansion model with 80 output channels in the second layer.
    """
    assert not pretrained
    return AI85NetExpansion(planes=80, **kwargs)


class AI85Net6(nn.Module):
    """
    5-Layer CNN for AI85
    """
    def __init__(self, num_classes=10, num_channels=3, dimensions=(28, 28),
                 planes=60, pool=2, fc_inputs=12, bias=False, **kwargs):
        super().__init__()

        # AI85 Limits
        assert planes + num_channels <= ai8x.dev.WEIGHT_INPUTS
        assert planes + fc_inputs <= ai8x.dev.WEIGHT_DEPTH-1
        assert dimensions[0] == dimensions[1]  # Only square supported

        # Keep track of image dimensions so one constructor works for all image sizes
        dim = dimensions[0]

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, planes, 3,
                                          padding=1, bias=bias, **kwargs)
        # padding 1 -> no change in dimensions -> MNIST: 28x28 | CIFAR: 32x32

        pad = 2 if dim == 28 else 1
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(planes, planes, 3, pool_size=2, pool_stride=2,
                                                 padding=pad, bias=bias, **kwargs)
        dim //= 2  # pooling, padding 0 -> MNIST: 14x14 | CIFAR: 16x16
        if pad == 2:
            dim += 2  # MNIST: padding 2 -> 16x16 | CIFAR: padding 1 -> 16x16

        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(planes, ai8x.dev.WEIGHT_DEPTH-planes-fc_inputs, 3,
                                                 pool_size=2, pool_stride=2, padding=1,
                                                 bias=bias, **kwargs)
        dim //= 2  # pooling, padding 0 -> 8x8
        # padding 1 -> no change in dimensions

        self.conv4 = ai8x.FusedAvgPoolConv2dReLU(ai8x.dev.WEIGHT_DEPTH-planes-fc_inputs,
                                                 fc_inputs, 3,
                                                 pool_size=pool, pool_stride=2, padding=1,
                                                 bias=bias, **kwargs)
        dim //= pool  # pooling, padding 0 -> 4x4
        # padding 1 -> no change in dimensions

        self.conv5 = ai8x.Conv2d(fc_inputs * dim * dim, num_classes, 1,
                                 padding=0, bias=None, **kwargs)
        # 10x1x1

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)

        return x


def ai85net6(pretrained=False, **kwargs):
    """
    Constructs a AI84Net6 model.
    """
    assert not pretrained
    return AI85Net6(**kwargs)


class AI85SqueezeNet(nn.Module):
    """
    SqueezeNet for AI85.
    The last layer is implemented as a linear layer rather than a convolution
    layer as defined in th eoriginal paper.
    """
    def __init__(self, num_channels=3, num_classes=10, dimensions=(32, 32),
                 bias=False, **kwargs):
        super().__init__()
        dim1 = dimensions[0]
        dim2 = dimensions[1]
        # 3x32x32
        self.conv1 = ai8x.FusedMaxPoolConv2dReLU(in_channels=num_channels, out_channels=64,
                                                 kernel_size=3, padding=1, bias=bias, **kwargs)
        dim1 //= 2
        dim2 //= 2
        # 64x16x16
        self.fire1 = ai8x_blocks.Fire(in_planes=64, squeeze_planes=16, expand1x1_planes=64,
                                      expand3x3_planes=64, bias=bias, **kwargs)
        # 128x16x16
        self.fire2 = ai8x_blocks.Fire(in_planes=128, squeeze_planes=16, expand1x1_planes=64,
                                      expand3x3_planes=64, bias=bias, **kwargs)
        # 128x16x16
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # check if kernel size=3
        dim1 //= 2
        dim2 //= 2
        # 128x8x8
        self.fire3 = ai8x_blocks.Fire(in_planes=128, squeeze_planes=32, expand1x1_planes=128,
                                      expand3x3_planes=128, bias=bias, **kwargs)
        # 256x8x8
        self.fire4 = ai8x_blocks.Fire(in_planes=256, squeeze_planes=32, expand1x1_planes=128,
                                      expand3x3_planes=128, bias=bias, **kwargs)
        # 256x8x8
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # check if kernel size=3
        dim1 //= 2
        dim2 //= 2
        # 256x4x4
        self.fire5 = ai8x_blocks.Fire(in_planes=256, squeeze_planes=48, expand1x1_planes=192,
                                      expand3x3_planes=192, bias=bias, **kwargs)
        # 384x4x4
        self.fire6 = ai8x_blocks.Fire(in_planes=384, squeeze_planes=48, expand1x1_planes=192,
                                      expand3x3_planes=192, bias=bias, **kwargs)
        # 384x4x4
        self.fire7 = ai8x_blocks.Fire(in_planes=384, squeeze_planes=48, expand1x1_planes=256,
                                      expand3x3_planes=256, bias=bias, **kwargs)
        # 512x4x4
        self.fire8 = ai8x_blocks.Fire(in_planes=512, squeeze_planes=64, expand1x1_planes=256,
                                      expand3x3_planes=256, bias=bias, **kwargs)
        # 512x4x4
        # self.conv2 = ai8x.FusedAvgPoolConv2dReLU(in_channels=512, out_channels=num_classes,
        #                                          kernel_size=1, pool_size=4, pool_stride=4)
        self.fc = ai8x.SoftwareLinear(512*dim1*dim2, num_classes, bias=bias)
        # num_classesx1x1

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.fire1(x)
        x = self.fire2(x)
        x = self.pool1(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.pool2(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        # x = self.conv2(x)
        # x = x.view(x.size(0), -1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ai85squeezenet(pretrained=False, **kwargs):
    """
    Constructs a AI85SqueezeNet model.
    """
    assert not pretrained
    return AI85SqueezeNet(**kwargs)


models = [
    {
        'name': 'ai85netwide',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85net80wide',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85net80expansion',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85net6',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'ai85squeezenet',
        'min_input': 1,
        'dim': 2,
    },
]
