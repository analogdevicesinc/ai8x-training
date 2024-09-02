###################################################################################################
#
# Copyright (C) 2021-2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
ImageNet EfficientNet v.2 network implementation for MAX78002.
"""
import math

from torch import nn

import ai8x
import ai8x_blocks


class AI87ImageNetEfficientNetV2(nn.Module):
    """
    EfficientNet v2 Model
    """
    def __init__(
            self,
            num_classes=1000,
            num_channels=3,
            dimensions=(112, 112),  # pylint: disable=unused-argument
            bias=False,
            **kwargs
    ):
        super().__init__()

        # Stem Layer
        self.conv_stem = ai8x.FusedMaxPoolConv2dBNReLU(num_channels, 32, 3, pool_size=2,
                                                       pool_stride=2, stride=1, batchnorm='Affine',
                                                       padding=1, bias=bias, eps=1e-03,
                                                       momentum=0.01, **kwargs)
        # 56x56
        # Series of MBConv blocks
        self.mb_conv1 = ai8x_blocks.MBConvBlock(32, 16, 3, bias=bias, se_ratio=None,
                                                expand_ratio=1, fused=True, **kwargs)
        self.mb_conv2 = ai8x_blocks.MBConvBlock(16, 32, 3, bias=bias, se_ratio=None,
                                                expand_ratio=4, fused=True, **kwargs)
        self.mb_conv3 = ai8x_blocks.MBConvBlock(32, 32, 3, bias=bias, se_ratio=None,
                                                expand_ratio=4, fused=True, **kwargs)
        self.mb_conv4 = ai8x_blocks.MBConvBlock(32, 48, 3, bias=bias, se_ratio=None,
                                                expand_ratio=4, fused=True, **kwargs)
        self.mb_conv5 = ai8x_blocks.MBConvBlock(48, 48, 3, bias=bias, se_ratio=None,
                                                expand_ratio=4, fused=True, **kwargs)
        self.mb_conv6 = ai8x_blocks.MBConvBlock(48, 48, 3, bias=bias, se_ratio=None,
                                                expand_ratio=4, fused=True, **kwargs)
        self.max_pooling1 = ai8x.MaxPool2d((2, 2))
        # 28x28
        self.mb_conv7 = ai8x_blocks.MBConvBlock(48, 96, 3, bias=bias, se_ratio=None,
                                                expand_ratio=2, fused=False, **kwargs)
        self.mb_conv8 = ai8x_blocks.MBConvBlock(96, 96, 3, bias=bias, se_ratio=None,
                                                expand_ratio=2, fused=False, **kwargs)
        self.mb_conv9 = ai8x_blocks.MBConvBlock(96, 128, 3, bias=bias, se_ratio=None,
                                                expand_ratio=2, fused=False, **kwargs)

        self.conv10 = ai8x.FusedMaxPoolConv2dBNReLU(128, 128, 1, pool_size=2,
                                                    pool_stride=2, stride=1, batchnorm='Affine',
                                                    padding=0, bias=bias, eps=1e-03,
                                                    momentum=0.01, **kwargs)
        # 14x14
        self.mb_conv11 = ai8x_blocks.MBConvBlock(128, 128, 3, bias=bias, se_ratio=None,
                                                 expand_ratio=4, fused=False, **kwargs)
        # Head Layer
        self.conv_head = ai8x.FusedConv2dBNReLU(128, 1024, 1, stride=1, batchnorm='Affine',
                                                padding=0, bias=bias, eps=1e-03,
                                                momentum=0.01, **kwargs)

        self.avg_pooling = ai8x.AvgPool2d((14, 14))
        # 1x1
        # self.dropout = nn.Dropout(0.2)
        # Final linear layer
        self.fc = ai8x.Linear(1024, num_classes, bias=bias, wide=True, **kwargs)

        self._initialize_weights()

    def forward(self, x):  # pylint: disable=arguments-differ
        """ Forward prop """
        x = self.conv_stem(x)
        x = self.mb_conv1(x)
        x = self.mb_conv2(x)
        x = self.mb_conv3(x)
        x = self.mb_conv4(x)
        x = self.mb_conv5(x)
        x = self.mb_conv6(x)
        x = self.max_pooling1(x)
        x = self.mb_conv7(x)
        x = self.mb_conv8(x)
        x = self.mb_conv9(x)
        x = self.conv10(x)
        x = self.mb_conv11(x)
        x = self.conv_head(x)
        x = self.avg_pooling(x)
        # x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / (n)))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def ai87imageneteffnetv2(pretrained=False, **kwargs):
    """
    Constructs EfficientNet v2 model.
    """
    assert not pretrained
    return AI87ImageNetEfficientNetV2(**kwargs)


models = [
    {
        'name': 'ai87imageneteffnetv2',
        'min_input': 1,
        'dim': 2,
    },
]
