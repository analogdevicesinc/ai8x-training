###################################################################################################
#
# Copyright (C) 2024 Analog Devices, Inc. All Rights Reserved.
# This software is proprietary to Analog Devices, Inc. and its licensors.
#
###################################################################################################
"""
Auto Encoder Network
"""

from torch import nn

import ai8x


class CNN_BASE(nn.Module):
    """
    Auto Encoder Network
    """
    def __init__(self,
                 num_channels=3,  # pylint: disable=unused-argument
                 bias=True,  # pylint: disable=unused-argument
                 weight_init="kaiming",  # pylint: disable=unused-argument
                 num_classes=0,  # pylint: disable=unused-argument
                 **kwargs):  # pylint: disable=unused-argument
        super().__init__()

    def initWeights(self, weight_init="kaiming"):
        """
        Auto Encoder Weight Initialization
        """
        weight_init = weight_init.lower()
        assert weight_init in ('kaiming', 'xavier', 'glorot')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if weight_init == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                elif weight_init in ('glorot', 'xavier'):
                    nn.init.xavier_uniform_(m.weight)

            elif isinstance(m, nn.ConvTranspose2d):
                if weight_init == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                elif weight_init in ('glorot', 'xavier'):
                    nn.init.xavier_uniform_(m.weight)

            elif isinstance(m, nn.Linear):
                if weight_init == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                elif weight_init in ('glorot', 'xavier'):
                    nn.init.xavier_uniform_(m.weight)


class AI85AutoEncoder(CNN_BASE):
    """
    Neural Network that has depthwise convolutions to reduce input dimensions.
    Filters work across individual axis data first.
    Output of 1D Conv layer is then flattened before being fed to fully connected layers
    Fully connected layers down sample the data to a bottleneck. This completes the encoder.
    The decoder is then the same in reverse

    Input Shape: [BATCH_SZ, FFT_LEN, N_AXES] -> [BATCH_SZ, 256, 3] = [N, N_CHANNELS, SIGNAL_LEN]
    """

    def __init__(self,
                 num_channels=256,
                 dimensions=None,  # pylint: disable=unused-argument
                 num_classes=1,  # pylint: disable=unused-argument
                 n_axes=3,
                 bias=True,
                 weight_init="kaiming",
                 batchNorm=True,
                 bottleNeckDim=4,
                 **kwargs):

        super().__init__()

        print("Batchnorm setting in model = ", batchNorm)

        weight_init = weight_init.lower()
        assert weight_init in ('kaiming', 'xavier', 'glorot')

        # Num channels is equal to the length of FFTs here
        self.num_channels = num_channels
        self.n_axes = n_axes

        S = 1
        P = 0

        # ----- DECODER ----- #
        # Kernel in 1st layer looks at 1 axis at a time. Output width = input width
        n_in = num_channels
        n_out = 128
        if batchNorm:
            self.en_conv1 = ai8x.FusedConv1dBNReLU(n_in, n_out, 1, stride=S, padding=P, dilation=1,
                                                   bias=bias, batchnorm='Affine', **kwargs)
        else:
            self.en_conv1 = ai8x.FusedConv1dReLU(n_in, n_out, 1, stride=S, padding=P, dilation=1,
                                                 bias=bias, **kwargs)
        self.layer1_n_in = n_in
        self.layer1_n_out = n_out

        # Kernel in 2nd layer looks at 3 axes at once. Output Width = 1. Depth=n_out
        n_in = n_out
        n_out = 64
        if batchNorm:
            self.en_conv2 = ai8x.FusedConv1dBNReLU(n_in, n_out, 3, stride=S, padding=P, dilation=1,
                                                   bias=bias, batchnorm='Affine', **kwargs)
        else:
            self.en_conv2 = ai8x.FusedConv1dReLU(n_in, n_out, 3, stride=S, padding=P, dilation=1,
                                                 bias=bias, **kwargs)
        self.layer2_n_in = n_in
        self.layer2_n_out = n_out

        n_in = n_out
        n_out = 32
        self.en_lin1 = ai8x.FusedLinearReLU(n_in, n_out, bias=bias, **kwargs)
        # ----- END OF DECODER ----- #

        # ---- BOTTLENECK ---- #
        n_in = n_out
        self.bottleNeckDim = bottleNeckDim
        n_out = self.bottleNeckDim
        self.en_lin2 = ai8x.Linear(n_in, n_out, bias=0, **kwargs)
        # ---- END OF BOTTLENECK ---- #

        # ----- ENCODER ----- #
        n_in = n_out
        n_out = 32
        self.de_lin1 = ai8x.FusedLinearReLU(n_in, n_out, bias=bias, **kwargs)

        n_in = n_out
        n_out = 96
        self.de_lin2 = ai8x.FusedLinearReLU(n_in, n_out, bias=bias, **kwargs)

        n_in = n_out
        n_out = num_channels*n_axes
        self.out_lin = ai8x.Linear(n_in, n_out, bias=0, **kwargs)
        # ----- END OF ENCODER ----- #

        self.initWeights(weight_init)

    def forward(self, x):
        """Forward prop"""
        x = self.en_conv1(x)
        x = self.en_conv2(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.en_lin1(x)
        x = self.en_lin2(x)

        x = self.de_lin1(x)
        x = self.de_lin2(x)
        x = self.out_lin(x)
        x = x.view(x.shape[0], self.num_channels, self.n_axes)

        return x


def ai85autoencoder(pretrained=False, **kwargs):
    """
    Constructs an Autoencoder model
    """
    assert not pretrained
    return AI85AutoEncoder(**kwargs)


models = [
    {
        'name': 'ai85autoencoder',
        'min_input': 1,
        'dim': 1,
    }
]
