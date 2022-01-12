###################################################################################################
#
# Copyright (C) 2019-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
AI85 demonstration network
"""
from torch import nn

import ai8x


class AI85AfskNet(nn.Module):
    """
    AI85 1D audio frequency-shift keying demodulator CNN.
    """
    def __init__(self, num_classes=2, num_channels=1, dimensions=(22, 1),
                 fc_inputs=16, bias=False, **kwargs):
        super().__init__()

        dim1 = dimensions[0]
        self.mfcc_conv1 = ai8x.FusedConv1dReLU(num_channels, 64, 5, stride=1,
                                               padding=2, bias=bias, **kwargs)
        self.dropout1 = nn.Dropout(0.2)
        self.mfcc_conv2 = ai8x.FusedConv1dReLU(64, 32, 5, stride=1, padding=2, bias=bias, **kwargs)
        self.mfcc_conv4 = ai8x.FusedConv1dReLU(32, fc_inputs, 5, stride=1,
                                               padding=2, bias=bias, **kwargs)
        self.fc = ai8x.Linear(fc_inputs * dim1, num_classes, bias=bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.mfcc_conv1(x)
        x = self.dropout1(x)
        x = self.mfcc_conv2(x)
        x = self.mfcc_conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ai85afsknet(pretrained=False, **kwargs):
    """
    Constructs a AI85AfskNet model.
    """
    assert not pretrained
    return AI85AfskNet(**kwargs)


models = [
    {
        'name': 'ai85afsknet',
        'min_input': 1,
        'dim': 1,
    },
]
