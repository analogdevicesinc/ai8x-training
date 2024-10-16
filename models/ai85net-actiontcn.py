###################################################################################################
#
# Copyright (C) 2022-2024 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Action recognition network for AI85
"""
import torch
from torch import nn

import ai8x


class AI85ActionTCN(nn.Module):
    """
    Conv2D backbone + TCN layers for Action Recognition
    Model was designed to be used with the Kinetics dataset.
    Number of frames was set to 15, as the model optimally performs with this number
    within the constraints of the AI85 hardware.
    """
    def __init__(
            self,
            num_classes=5,
            dimensions=(60, 60),  # pylint: disable=unused-argument
            num_channels=96,
            bias=True,
            bn='Affine',
            dropout=0.5,
            **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cnn_out_shape = (1, 1)
        self.cnn_out_channel = 32
        self.num_frames = 15
        num_filters = 64
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel

        self.prep0 = ai8x.FusedConv2dBNReLU(num_channels, num_filters, 1, stride=1,
                                            padding=0, bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv0 = ai8x.FusedConv2dBNReLU(num_filters, num_filters, 3, padding=1,
                                            bias=bias, batchnorm=bn, **kwargs)
        self.conv1 = ai8x.FusedMaxPoolConv2dBNReLU(num_filters, num_filters, 3, padding=1,
                                                   bias=bias, batchnorm=bn, **kwargs)
        self.conv1_2 = ai8x.FusedConv2dBNReLU(num_filters, num_filters, 3, padding=1,
                                              bias=bias, batchnorm=bn, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dBNReLU(num_filters, num_filters, 3, padding=1,
                                                   bias=bias, batchnorm=bn, **kwargs)
        self.conv2_1 = ai8x.FusedConv2dBNReLU(num_filters, num_filters, 1, padding=0,
                                              bias=bias, batchnorm=bn, **kwargs)
        self.conv2_p = ai8x.FusedMaxPoolConv2dBNReLU(num_filters, num_filters, 3, padding=1,
                                                     bias=bias, batchnorm=bn, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(num_filters, num_filters, 3, padding=1,
                                                   bias=bias, batchnorm=bn, **kwargs)
        self.conv3_1 = ai8x.FusedConv2dBNReLU(num_filters, num_filters, 1, padding=0,
                                              bias=bias, batchnorm=bn, **kwargs)
        self.conv3_p = ai8x.FusedMaxPoolConv2dBNReLU(num_filters, num_filters, 3, padding=1,
                                                     bias=bias, batchnorm=bn, **kwargs)
        self.conv4 = ai8x.FusedMaxPoolConv2dBNReLU(num_filters, num_filters, 3, padding=1,
                                                   bias=bias, batchnorm=bn, **kwargs)
        self.conv4_1 = ai8x.FusedConv2dBNReLU(num_filters, num_filters, 1, padding=0,
                                              bias=bias, batchnorm=bn, **kwargs)
        self.conv4_p = ai8x.FusedMaxPoolConv2dBNReLU(num_filters, num_filters, 3, padding=1,
                                                     bias=bias, batchnorm=bn, **kwargs)
        self.conv5 = ai8x.FusedConv2dBNReLU(num_filters, self.cnn_out_channel, 3, padding=0,
                                            bias=bias, batchnorm=bn, **kwargs)

        self.drop2 = nn.Dropout2d(dropout)
        self.drop3 = nn.Dropout2d(dropout)
        self.drop4 = nn.Dropout2d(dropout)

        self.add = ai8x.Add()

        self.tcn0 = ai8x.FusedConv1dBNReLU(len_frame_vector, len_frame_vector, 3, padding=0,
                                           stride=1, dilation=1, bias=bias, batchnorm=bn, **kwargs)
        self.tcn1 = ai8x.FusedConv1dBNReLU(len_frame_vector, len_frame_vector, 3, padding=0,
                                           stride=1, dilation=2, bias=bias, batchnorm=bn, **kwargs)
        self.tcn2 = ai8x.Conv1d(len_frame_vector, num_classes, 3, wide=True, padding=0,
                                stride=1, dilation=4, bias=bias, batchnorm=bn, **kwargs)

    def create_prep(self, x):
        """Prep layer(s)"""
        c = self.prep0(x)
        return c

    def create_cnn(self, x):
        """2D CNN backbone"""
        cx = self.conv0(x)

        c = self.conv1(cx)
        cx = self.conv1_2(c)

        c = self.conv2(cx)
        c = self.conv2_1(c)
        cp = self.conv2_p(cx)
        cx = self.add(cp, self.drop2(c))

        c = self.conv3(cx)
        c = self.conv3_1(c)
        cp = self.conv3_p(cx)
        cx = self.add(cp, self.drop3(c))

        c = self.conv4(cx)
        c = self.conv4_1(c)
        cp = self.conv4_p(cx)
        cx = self.add(cp, self.drop4(c))

        c = self.conv5(cx)

        return c

    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        cnnoutputs = torch.zeros_like(x)
        cnnoutputs = cnnoutputs[:, :, :self.cnn_out_channel, :self.cnn_out_shape[0],
                                :self.cnn_out_shape[1]]

        for i in range(self.num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs = assign_cnnoutputs(cnnoutputs, i, self.create_cnn(prep_out))
        tcn_input = cnnoutputs.permute(0, 1, 3, 4, 2).reshape(batch_size, self.num_frames, -1) \
                                                     .permute(0, 2, 1)
        tcn_output = self.tcn0(tcn_input)
        tcn_output = self.tcn1(tcn_output)
        tcn_output = self.tcn2(tcn_output)

        return tcn_output.reshape(batch_size, self.num_classes)


@torch.fx.wrap
def assign_cnnoutputs(cnnoutputs, index, value):
    """
    Assigns a value to a slice of a tensor, required for symbolic tracing
    """
    cnnoutputs[:, index] = value
    return cnnoutputs


def ai85actiontcn(pretrained=False, **kwargs):
    """
    Constructs an AI85ActionTCN model.
    rn AI85Action(**kwargs)
    """
    assert not pretrained
    return AI85ActionTCN(**kwargs)


models = [
    {
        'name': 'ai85actiontcn',
        'min_input': 1,
        'dim': 1,
    },
]
