###################################################################################################
#
# Copyright (C) 2020-2024 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Contains implementations of popular neural network blocks by taking MAX7800X limits into account.
"""

import torch
from torch import nn
from torch.nn import functional as F

import ai8x


class Fire(nn.Module):
    """
    AI8X - Fire Layer
    """
    def __init__(self, in_planes, squeeze_planes, expand1x1_planes, expand3x3_planes,
                 bias=True, **kwargs):
        super().__init__()
        self.squeeze_layer = ai8x.FusedConv2dReLU(in_channels=in_planes,
                                                  out_channels=squeeze_planes, kernel_size=1,
                                                  bias=bias, **kwargs)
        self.expand1x1_layer = ai8x.FusedConv2dReLU(in_channels=squeeze_planes,
                                                    out_channels=expand1x1_planes, kernel_size=1,
                                                    bias=bias, **kwargs)
        self.expand3x3_layer = ai8x.FusedConv2dReLU(in_channels=squeeze_planes,
                                                    out_channels=expand3x3_planes, kernel_size=3,
                                                    padding=1, bias=bias, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.squeeze_layer(x)
        return torch.cat([self.expand1x1_layer(x), self.expand3x3_layer(x)], 1)


class ResidualBottleneck(nn.Module):
    """
    AI8X - Residual Bottleneck Layer.
    This module uses ReLU activation not ReLU6 as the original study suggests [1],
    because of MAX7800X capabilities.

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        expansion_factor: expansion_factor
        stride: stirde size (default=1)
        bias: determines if bias used at non-depthwise layers.
        depthwise_bias: determines if bias used at depthwise layers.

    References:
        [1] https://arxiv.org/pdf/1801.04381.pdf (MobileNetV2)
    """
    def __init__(self, in_channels, out_channels, expansion_factor, stride=1, bias=False,
                 depthwise_bias=False, **kwargs):
        super().__init__()
        self.stride = stride
        hidden_channels = int(round(in_channels * expansion_factor))
        if hidden_channels == in_channels:
            self.conv1 = ai8x.Empty()
        else:
            self.conv1 = ai8x.FusedConv2dBNReLU(in_channels, hidden_channels, 1, padding=0,
                                                bias=bias, **kwargs)
        if stride == 1:
            if depthwise_bias:
                self.conv2 = ai8x.FusedDepthwiseConv2dBNReLU(hidden_channels, hidden_channels, 3,
                                                             padding=1, stride=stride,
                                                             bias=depthwise_bias, **kwargs)
            else:
                self.conv2 = ai8x.FusedDepthwiseConv2dReLU(hidden_channels, hidden_channels, 3,
                                                           padding=1, stride=stride,
                                                           bias=depthwise_bias, **kwargs)

        else:
            if depthwise_bias:
                self.conv2 = ai8x.FusedMaxPoolDepthwiseConv2dBNReLU(hidden_channels,
                                                                    hidden_channels,
                                                                    3, padding=1, pool_size=stride,
                                                                    pool_stride=stride,
                                                                    bias=depthwise_bias,
                                                                    **kwargs)
            else:
                self.conv2 = ai8x.FusedMaxPoolDepthwiseConv2dReLU(hidden_channels,
                                                                  hidden_channels,
                                                                  3, padding=1, pool_size=stride,
                                                                  pool_stride=stride,
                                                                  bias=depthwise_bias,
                                                                  **kwargs)

        self.conv3 = ai8x.FusedConv2dBN(hidden_channels, out_channels, 1, bias=bias, **kwargs)

        if (stride == 1) and (in_channels == out_channels):
            self.resid = ai8x.Add()
        else:
            self.resid = self.NoResidual()

    class NoResidual(nn.Module):
        """
        Does nothing.
        """
        def forward(self, *x):  # pylint: disable=arguments-differ
            """Forward prop"""
            return x[0]

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        return self.resid(y, x)


class ConvResidualBottleneck(nn.Module):
    """
    AI8X module based on Residual Bottleneck Layer.
    Depthwise convolution is replaced with standard convolution.
    This module uses ReLU activation not ReLU6 as the original study suggests [1],
    because of MAX7800X capabilities.

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        expansion_factor: expansion_factor
        stride: stirde size (default=1)
        bias: determines if bias used at non-depthwise layers.
        depthwise_bias: determines if bias used at depthwise layers.

    References:
        [1] https://arxiv.org/pdf/1801.04381.pdf (MobileNetV2)
    """
    def __init__(self, in_channels, out_channels, expansion_factor, stride=1, bias=False,
                 depthwise_bias=False, **kwargs):
        super().__init__()
        self.stride = stride
        hidden_channels = int(round(in_channels * expansion_factor))
        if hidden_channels == in_channels:
            self.conv1 = ai8x.Empty()
        else:
            self.conv1 = ai8x.FusedConv2dBNReLU(in_channels, hidden_channels, 1, padding=0,
                                                bias=bias, **kwargs)
        if stride == 1:
            if depthwise_bias:
                self.conv2 = ai8x.FusedConv2dBN(hidden_channels, out_channels, 3,
                                                padding=1, stride=stride,
                                                bias=depthwise_bias, **kwargs)

            else:
                self.conv2 = ai8x.Conv2d(hidden_channels, out_channels, 3,
                                         padding=1, stride=stride,
                                         bias=depthwise_bias, **kwargs)

        else:
            if depthwise_bias:
                self.conv2 = ai8x.FusedMaxPoolConv2dBN(hidden_channels,
                                                       out_channels, 3,
                                                       padding=1, pool_size=stride,
                                                       pool_stride=stride,
                                                       bias=depthwise_bias, **kwargs)

            else:
                self.conv2 = ai8x.FusedMaxPoolConv2d(hidden_channels,
                                                     out_channels, 3,
                                                     padding=1, pool_size=stride,
                                                     pool_stride=stride,
                                                     bias=depthwise_bias, **kwargs)

        if (stride == 1) and (in_channels == out_channels):
            self.resid = ai8x.Add()
        else:
            self.resid = self.NoResidual()

    class NoResidual(nn.Module):
        """
        Does nothing.
        """
        def forward(self, *x):  # pylint: disable=arguments-differ
            """Forward prop"""
            return x[0]

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        y = self.conv1(x)
        y = self.conv2(y)
        return self.resid(y, x)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        image_size (tuple or list): [image_height, image_width].
        in_channels: number of input channels
        out_channels: number of output channels
        kernel_size: kernel size (default 3)
        stride: stride size (default 1)
        se_ratio: squeeze and excitation (SE) ratio (0-1)
        expand_ratio: expansion ratio (default 1)
        fused: eliminates depthwise convolution layer

    References:
        [1] https://arxiv.org/pdf/2104.00298.pdf (EfficientNetV2)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 bias=False,
                 se_ratio=None,
                 expand_ratio=1,
                 fused=False,
                 **kwargs):
        super().__init__()

        self.has_se = (se_ratio is not None) and (0 < se_ratio <= 1)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.fused = fused

        # Expansion phase (Inverted Bottleneck)
        inp = in_channels  # number of input channels
        out = in_channels * expand_ratio  # number of output channels
        if expand_ratio != 1:
            if fused is True:
                self.expand_conv = ai8x.FusedConv2dBNReLU(inp, out, kernel_size=kernel_size,
                                                          padding=1, batchnorm='Affine', bias=bias,
                                                          eps=1e-03, momentum=0.01, **kwargs)
            else:
                self.expand_conv = ai8x.FusedConv2dBNReLU(inp, out, 1,
                                                          batchnorm='Affine', bias=bias,
                                                          eps=1e-03, momentum=0.01, **kwargs)
        # Depthwise Convolution phase
        if fused is not True:
            self.depthwise_conv = ai8x.FusedDepthwiseConv2dBNReLU(out, out, kernel_size,
                                                                  padding=1, stride=stride,
                                                                  batchnorm='Affine', bias=bias,
                                                                  eps=1e-03, momentum=0.01,
                                                                  **kwargs)
        # Squeeze and Excitation phase
        if self.has_se:
            num_squeezed_channels = max(1, int(in_channels * se_ratio))
            self.se_reduce = ai8x.FusedConv2dReLU(in_channels=out,
                                                  out_channels=num_squeezed_channels,
                                                  kernel_size=1, stride=1, bias=bias, **kwargs)
            self.se_expand = ai8x.Conv2d(in_channels=num_squeezed_channels, out_channels=out,
                                         kernel_size=1, stride=1, bias=bias, **kwargs)
        # Output Convolution phase
        final_out = out_channels
        self.project_conv = ai8x.FusedConv2dBN(in_channels=out, out_channels=final_out,
                                               kernel_size=1, batchnorm='Affine', bias=bias,
                                               eps=1e-03, momentum=0.01, **kwargs)
        # Skip connection
        input_filters, output_filters = self.in_channels, self.out_channels
        if self.stride == 1 and input_filters == output_filters:
            self.resid = ai8x.Add()

    def forward(self, inputs):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of this block after processing.
        """
        # Expansion Convolution layer
        x = inputs
        if self.expand_ratio != 1:
            x = self.expand_conv(inputs)
        # Depthwise Convolution layer
        if self.fused is not True:
            x = self.depthwise_conv(x)
        # Squeeze and Excitation layers
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self.se_reduce(x_squeezed)
            x_squeezed = self.se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x
        # Output Convolution layer
        x = self.project_conv(x)
        # Skip connection
        input_filters, output_filters = self.in_channels, self.out_channels
        if self.stride == 1 and input_filters == output_filters:
            x = self.resid(x, inputs)
        return x
