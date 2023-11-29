###################################################################################################
#
#MIT License

#Copyright (c) 2019 Jian Zhao

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#
###################################################################################################
#
# Portions Copyright (C) 2023 Maxim Integrated Products, Inc.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple
import torch.nn.functional as F
import torchvision.transforms.functional as FT

# Support: ['IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']

class DRL(nn.Module):
    """
    Dimensionality reduction layers
    Expects unnormalized 512 embeddings from the Teacher Model
    """
    def __init__(
            self,
            dimensionality,
            bias=True,
            **kwargs
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(512, 512, 1, padding=0, bias=bias, **kwargs)
        self.BN1 =  nn.BatchNorm1d(512)
        self.PRelu1 = nn.PReLU(512)
        self.conv2 = nn.Conv1d(512, dimensionality, 1, padding=0, bias=bias, **kwargs)
        self.BN2 =  nn.BatchNorm1d(dimensionality)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = torch.unsqueeze(x, 2)
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.PRelu1(x)
        x = self.conv2(x)
        x = self.BN2(x)
        x = torch.squeeze(x, 2)
        return x



class Ensemble(nn.Module):
    def __init__(self, resnet, DRL):
        super().__init__()
        self.resnet = resnet
        self.DRL = DRL
        self.Teacher_mode = False

    def forward(self, x):
        if x.shape[1] == 6:
            if (not self.Teacher_mode):
                self.Teacher_mode=True
            x = x[:,3: ,:,:]
            x_flip = FT.hflip(x)
            x = torch.cat((x, x_flip), 0)
        x = self.resnet(x)
        x = self.DRL(x)
        if self.Teacher_mode:
            x = x[:x.shape[0]//2] + x[x.shape[0]//2:] #Flip fusion
        x = F.normalize(x, p=2, dim=1)
        return x

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    return blocks


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir'):
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        if input_size[0] == 112:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(p=0), # Dropout is set to 0, due to the train.py structure
                                           Flatten(),
                                           Linear(512 * 7 * 7, 512),
                                           BatchNorm1d(512))
        else:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(p=0),
                                           Flatten(),
                                           Linear(512 * 14 * 14, 512),
                                           BatchNorm1d(512))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self._initialize_weights()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        #x = F.normalize(x, p=2, dim=1) Don't normalize in the backbone for DRL

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


def ir_50(input_size=[112,112], dimensionality=64, backbone_checkpoint = None, **kwargs):
    """Constructs a ir-50 model.
    """
    model = Backbone(input_size, 50, 'ir')
    if backbone_checkpoint is not None:
        model.load_state_dict(torch.load(backbone_checkpoint, map_location=torch.device('cpu')))
    for param in model.parameters():
        param.requires_grad = False
    drl = DRL(dimensionality)
    ensemble = Ensemble(model, drl)

    return ensemble


def ir_101(input_size=[112,112], dimensionality=64, backbone_checkpoint = None, **kwargs):
    """Constructs a ir-101 model.
    """
    model = Backbone(input_size, 100, 'ir')
    if backbone_checkpoint is not None:
        model.load_state_dict(torch.load(backbone_checkpoint, map_location=torch.device('cpu')))
    for param in model.parameters():
        param.requires_grad = False
    drl = DRL(dimensionality)
    ensemble = Ensemble(model, drl)

    return ensemble


def ir_152(input_size=[112,112], dimensionality=64, backbone_checkpoint = None, **kwargs):
    """Constructs a ir-152 model.
    """
    model = Backbone(input_size, 152, 'ir')
    if backbone_checkpoint is not None:
        model.load_state_dict(torch.load(backbone_checkpoint, map_location=torch.device('cpu')))
    for param in model.parameters():
        param.requires_grad = False
    drl = DRL(dimensionality)

    ensemble = Ensemble(model, drl)

    return ensemble


def ir_se_50(input_size=[112,112], dimensionality=64, backbone_checkpoint = None, **kwargs):
    """Constructs a ir_se-50 model.
    """
    model = Backbone(input_size, 50, 'ir_se')
    if backbone_checkpoint is not None:
        model.load_state_dict(torch.load(backbone_checkpoint, map_location=torch.device('cpu')))
    for param in model.parameters():
        param.requires_grad = False
    drl = DRL(dimensionality)
    ensemble = Ensemble(model, drl)

    return ensemble


def ir_se_101(input_size=[112,112], dimensionality=64, backbone_checkpoint = None, **kwargs):
    """Constructs a ir_se-101 model.
    """
    model = Backbone(input_size, 100, 'ir_se')
    if backbone_checkpoint is not None:
        model.load_state_dict(torch.load(backbone_checkpoint, map_location=torch.device('cpu')))
    for param in model.parameters():
        param.requires_grad = False
    drl = DRL(dimensionality)
    ensemble = Ensemble(model, drl)

    return ensemble


def ir_se_152(input_size=[112,112], dimensionality=64, backbone_checkpoint = None, **kwargs):
    """Constructs a ir_se-152 model.
    """
    model = Backbone(input_size, 152, 'ir_se')
    if backbone_checkpoint is not None:
        model.load_state_dict(torch.load(backbone_checkpoint, map_location=torch.device('cpu')))
    for param in model.parameters():
        param.requires_grad = False
    drl = DRL(dimensionality)
    ensemble = Ensemble(model, drl)

    return ensemble

models = [
    {
        'name': 'ir_50',
        'min_input': 1,
        'dim': 2,
        'dr': True,
    },
    {
        'name': 'ir_101',
        'min_input': 1,
        'dim': 2,
        'dr': True,
    },
    {
        'name': 'ir_152',
        'min_input': 1,
        'dim': 2,
        'dr': True,
    },
    {
        'name': 'ir_se_50',
        'min_input': 1,
        'dim': 2,
        'dr': True,
    },
    {
        'name': 'ir_se_101',
        'min_input': 1,
        'dim': 2,
        'dr': True,
    },
    {
        'name': 'ir_se_152',
        'min_input': 1,
        'dim': 2,
        'dr': True,
    },

]