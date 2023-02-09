###################################################################################################
#
# Copyright (C) 2022-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Action recognition network for AI85
"""
from torch import nn
import torch
import ai8x

class AI85ActionBackbone(nn.Module):
    """
    Conv2D backbone for Action Recognition
    """
    def __init__(
            self,
            num_classes=5,
            dimensions=(240,240),  # pylint: disable=unused-argument
            num_channels=6,
            fold_ratio=4,
            bias=True,
            bn='Affine',
            dropout=0.5,
            **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_final_channels = num_channels*fold_ratio*fold_ratio
        self.bias = bias
        self.bn = bn
        self.cnn_out_shape = (1, 1)
        self.cnn_out_channel = 32
        self.p = dropout
        num_filters = 64
        len_frame_vector = self.cnn_out_shape[0]*self.cnn_out_shape[1]*self.cnn_out_channel
        kwargs = {}

        self.prep0 = ai8x.FusedConv2dBNReLU(self.num_final_channels, num_filters, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.conv0 = ai8x.FusedConv2dBNReLU(num_filters, num_filters, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1 = ai8x.FusedMaxPoolConv2dBNReLU(num_filters, num_filters, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_1 = ai8x.FusedConv2dBNReLU(num_filters, num_filters, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv1_p = ai8x.FusedMaxPoolConv2dBNReLU(num_filters, num_filters, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dBNReLU(num_filters, num_filters, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_1 = ai8x.FusedConv2dBNReLU(num_filters, num_filters, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv2_p = ai8x.FusedMaxPoolConv2dBNReLU(num_filters, num_filters, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(num_filters, num_filters, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_1 = ai8x.FusedConv2dBNReLU(num_filters, num_filters, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv3_p = ai8x.FusedMaxPoolConv2dBNReLU(num_filters, num_filters, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv4 = ai8x.FusedMaxPoolConv2dBNReLU(num_filters, num_filters, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv4_1 = ai8x.FusedConv2dBNReLU(num_filters, num_filters, 1, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv4_p = ai8x.FusedMaxPoolConv2dBNReLU(num_filters, num_filters, 3, padding=1, bias=bias, batchnorm=self.bn, **kwargs)
        self.conv5 = ai8x.FusedConv2dBNReLU(num_filters, self.cnn_out_channel, 3, padding=0, bias=bias, batchnorm=self.bn, **kwargs)
        
        self.drop1 = nn.Dropout2d(self.p)
        self.drop2 = nn.Dropout2d(self.p)
        self.drop3 = nn.Dropout2d(self.p)
        self.drop4 = nn.Dropout2d(self.p)

        self.fc = ai8x.Linear(len_frame_vector, num_classes, wide=True, bias=False, **kwargs)
        
    def create_prep(self, x):
        c = self.prep0(x)
        return c
    
    def create_cnn(self, x):
        cx = self.conv0(x)
        
        c = self.conv1(cx)
        c = self.conv1_1(c)
        cp = self.conv1_p(cx)
        cx = cp + self.drop1(c)
        
        c = self.conv2(cx)
        c = self.conv2_1(c)
        cp = self.conv2_p(cx)
        cx = cp + self.drop2(c)

        c = self.conv3(cx)
        c = self.conv3_1(c)
        cp = self.conv3_p(cx)
        cx = cp + self.drop3(c)

        c = self.conv4(cx)
        c = self.conv4_1(c)
        cp = self.conv4_p(cx)
        cx = cp + self.drop4(c)
        
        c = self.conv5(cx)
                
        return c

    def forward(self, x):
        """Forward prop"""
        batch_size = x.shape[0]
        num_frames = x.shape[1]
        cnnoutputs = torch.zeros(batch_size, num_frames, self.cnn_out_channel, self.cnn_out_shape[0], self.cnn_out_shape[1]).to(x.get_device())
        for i in range(num_frames):
            prep_out = self.create_prep(x[:, i])
            cnnoutputs[:, i] = self.create_cnn(prep_out)
        
        linear_input = cnnoutputs.permute(0, 1, 3, 4, 2).reshape(batch_size, num_frames, -1)

        outputs = torch.zeros((batch_size, self.num_classes)).to(x.get_device())   
        for i in range(num_frames):
            outputs += self.fc(linear_input[:, i]) / num_frames

        return outputs


def ai85actionbackbone(pretrained=False, **kwargs):
    """
    Constructs an AI85ActionBackbone model.
    rn AI85Action(**kwargs)
    """
    assert not pretrained
    return AI85ActionBackbone(**kwargs)

models = [
    {
        'name': 'ai85actionbackbone',
        'min_input': 1,
        'dim': 1,
    },
]