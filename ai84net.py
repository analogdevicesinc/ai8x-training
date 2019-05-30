###################################################################################################
#
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
###################################################################################################
"""
Network(s) that fit into AI84

Optionally quantize/clamp activations
"""
from torch.autograd import Function
import torch.nn as nn
import ai84


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


class AI84Net5(nn.Module):
    """
    CNN that uses max parameters in AI84
    """
    def __init__(self, num_classes=10, num_channels=3, dimensions=(28, 28),
                 quantize=False, clamp_range1=False,
                 planes=60, pool=4, fc_inputs=12, bias=False):
        super(AI84Net5, self).__init__()

        # AI84 Limits
        assert planes + num_channels <= ai84.WEIGHT_INPUTS
        assert planes + fc_inputs <= ai84.WEIGHT_DEPTH-1
        assert pool <= ai84.MAX_AVG_POOL
        assert pool & 1 == 0  # Only 0x0, 2x2 and 4x4 supported
        assert dimensions[0] == dimensions[1]  # Only square supported
        bits = ai84.ACTIVATION_BITS

        if quantize:
            self.quantize8 = Quantize(num_bits=bits)
            self.quantize_pool = Floor()
        else:
            self.quantize8 = Empty()
            self.quantize_pool = Empty()

        if clamp_range1:
            self.clamp = Clamp(min_val=-1., max_val=1.)  # Do not combine with ReLU
        elif quantize:
            self.clamp = Clamp(min_val=-(2**(bits-1)), max_val=2**(bits-1)-1)
        else:
            self.clamp = Empty()
        self.relu = nn.ReLU(inplace=True)

        # Keep track of image dimensions so one constructor works for all image sizes
        dim = dimensions[0]

        self.conv1 = nn.Conv2d(num_channels, planes, kernel_size=3,
                               stride=1, padding=2, bias=bias)
        dim += 2  # padding -> 30x30

        # MaxPool2d: stride and kernel_size must be the same
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1 if pool == 3 else 0)
        if pool != 3:
            dim -= 2  # stride of 2 -> 14x14, else 15x15
        dim //= 2
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1 if pool == 3 else 2, bias=bias)
        if pool != 3:
            dim += 2  # padding 2 -> 16x16, else 15x15
        self.conv3 = nn.Conv2d(planes, ai84.WEIGHT_DEPTH-planes-fc_inputs, kernel_size=3,
                               stride=1, padding=1, bias=bias)
        # no change in dimensions
        self.avgpool = nn.AvgPool2d(pool)
        dim //= pool  # pooling -> 4x4, else 3x3 or 5x5
        self.conv4 = nn.Conv2d(ai84.WEIGHT_DEPTH-planes-fc_inputs, fc_inputs, kernel_size=3,
                               stride=1, padding=1, bias=bias)
        # no change in dimensions
        self.fc = nn.Linear(fc_inputs*dim*dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.conv1(x)
        x = self.clamp(self.quantize8(self.relu(x)))
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.clamp(self.quantize8(self.relu(x)))
        x = self.conv3(x)
        x = self.clamp(self.quantize8(self.relu(x)))
        x = self.avgpool(x)
        x = self.clamp(self.quantize_pool(x))
        x = self.conv4(x)
        x = self.clamp(self.quantize8(self.relu(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ai84net5(pretrained=False, **kwargs):
    """
    Constructs a AI84Net-5 model.
    """
    assert not pretrained
    return AI84Net5(**kwargs)
