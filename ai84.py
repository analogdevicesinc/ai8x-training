###################################################################################################
#
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
###################################################################################################
"""
Contains the limits of the AI84 implementation and custom Pytorch modules that take
the limits into account.
"""
from torch.autograd import Function
import torch.nn as nn


WEIGHT_BITS = 8
ACTIVATION_BITS = 8

WEIGHT_INPUTS = 64
WEIGHT_DEPTH = 128

MAX_AVG_POOL = 4


class QuantizationFunction(Function):
    """
    Custom AI84 autograd function
    The forward pass quantizes to [-(2**(num_bits-1)), 2**(num_bits-1)-1].
    The backward pass is straight through.
    """
    @staticmethod
    def forward(ctx, x, bits=None):  # pylint: disable=arguments-differ
        return x.add(.5).div(2**(bits-1)).add(.5).floor()

    @staticmethod
    def backward(ctx, x):  # pylint: disable=arguments-differ
        # Straight through - return as many input gradients as there were arguments;
        # gradients of non-Tensor arguments to forward must be None.
        return x, None


class Quantize(nn.Module):
    """
    Post-activation integer quantization module
    Apply the custom autograd function
    """
    def __init__(self, num_bits=8):
        super(Quantize, self).__init__()
        self.num_bits = num_bits

    def forward(self, x):  # pylint: disable=arguments-differ
        return QuantizationFunction.apply(x, self.num_bits)


class FloorFunction(Function):
    """
    Custom AI84 autograd function
    The forward pass returns the integer floor.
    The backward pass is straight through.
    """
    @staticmethod
    def forward(ctx, x):  # pylint: disable=arguments-differ
        return x.floor()

    @staticmethod
    def backward(ctx, x):  # pylint: disable=arguments-differ
        # Straight through - return as many input gradients as there were arguments;
        # gradients of non-Tensor arguments to forward must be None.
        return x


class Floor(nn.Module):
    """
    Post-pooling integer quantization module
    Apply the custom autograd function
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        return FloorFunction.apply(x)


class Clamp(nn.Module):
    """
    Post-Activation Clamping Module
    Clamp the output to the given range
    """
    def __init__(self, min_val=None, max_val=None):
        super(Clamp, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):  # pylint: disable=arguments-differ
        return x.clamp(min=self.min_val, max=self.max_val)


class Empty(nn.Module):
    """
    Do nothing
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        return x


class FusedMaxPoolConv2dReLU(nn.Module):
    """
    AI84 - Fused 2D Max Pool, 2D Convolution and ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, pool_stride=2,
                 stride=1, padding=0, bias=True, relu=True, simulate=False):
        super(FusedMaxPoolConv2dReLU, self).__init__()

        assert pool_size & 1 == 0
        assert pool_size <= 16
        assert 0 < pool_stride <= 16
        assert 0 <= padding <= 2
        assert kernel_size == 3
        assert stride == 1

        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride, padding=0)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=bias)

        if simulate:
            bits = ACTIVATION_BITS
            self.quantize8 = Quantize(num_bits=bits)
            self.clamp = Clamp(min_val=-(2**(bits-1)), max_val=2**(bits-1)-1)
        else:
            self.quantize8 = Empty()
            self.clamp = Clamp(min_val=-1., max_val=1.)  # Do not combine with ReLU

        if relu:
            self.activate = nn.ReLU(inplace=True)
        else:
            self.activate = Empty()

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.pool(x)
        x = self.conv2d(x)
        x = self.clamp(self.quantize8(self.activate(x)))
        return x


class FusedMaxPoolConv2d(FusedMaxPoolConv2dReLU):
    """
    AI84 - Fused 2D Max Pool and 2D Convolution without activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(FusedMaxPoolConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                                 relu=False, **kwargs)


class FusedAvgPoolConv2dReLU(nn.Module):
    """
    AI84 - Fused 2D Avg Pool, 2D Convolution and ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, pool_stride=2,
                 stride=1, padding=0, bias=True, relu=True, simulate=False):
        super(FusedAvgPoolConv2dReLU, self).__init__()

        assert pool_size & 1 == 0
        assert pool_size <= 4
        assert 0 < pool_stride <= 4
        assert 0 <= padding <= 2
        assert kernel_size == 3
        assert stride == 1

        self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_stride, padding=0)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=bias)

        if simulate:
            bits = ACTIVATION_BITS
            self.quantize8 = Quantize(num_bits=bits)
            self.quantize_pool = Floor()
            self.clamp = Clamp(min_val=-(2**(bits-1)), max_val=2**(bits-1)-1)
        else:
            self.quantize8 = Empty()
            self.quantize_pool = Empty()
            self.clamp = Clamp(min_val=-1., max_val=1.)  # Do not combine with ReLU

        if relu:
            self.activate = nn.ReLU(inplace=True)
        else:
            self.activate = Empty()

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.clamp(self.quantize_pool(self.pool(x)))
        x = self.conv2d(x)
        x = self.clamp(self.quantize8(self.activate(x)))
        return x


class FusedAvgPoolConv2d(FusedAvgPoolConv2dReLU):
    """
    AI84 - Fused 2D Avg Pool and 2D Convolution without activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(FusedAvgPoolConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                                 relu=False, **kwargs)


class FusedConv2dReLU(nn.Module):
    """
    AI84 - Fused 2D Convolution and ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True,
                 relu=True, simulate=False):
        super(FusedConv2dReLU, self).__init__()

        assert 0 < stride <= 3
        assert 0 <= padding <= 2

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                                padding=padding, bias=bias)

        if simulate:
            bits = ACTIVATION_BITS
            self.quantize8 = Quantize(num_bits=bits)
            self.clamp = Clamp(min_val=-(2**(bits-1)), max_val=2**(bits-1)-1)
        else:
            self.quantize8 = Empty()
            self.clamp = Clamp(min_val=-1., max_val=1.)  # Do not combine with ReLU

        if relu:
            self.activate = nn.ReLU(inplace=True)
        else:
            self.activate = Empty()

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.conv2d(x)
        x = self.clamp(self.quantize8(self.activate(x)))
        return x


class Conv2d(FusedConv2dReLU):
    """
    AI84 - 2D Convolution without activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, relu=False, **kwargs)
