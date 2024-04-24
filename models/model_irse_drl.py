###################################################################################################
#
# Copyright (c) 2020 PaddlePaddle Authors.
# Portions Copyright (c) 2019 Jian Zhao
# Portions Copyright (C) 2023-2024 Maxim Integrated Products, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###################################################################################################
"""
FaceID Teacher Model to be used for Knowledge Distillation
See https://github.com/analogdevicesinc/ai8x-training/blob/develop/docs/FacialRecognitionSystem.md
"""
import sys
from collections import namedtuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as FT
from torch import nn


class DRL(nn.Module):
    """
    Dimensionality reduction layers
    Expects unnormalized 512 embeddings from the Teacher Model
    """
    def __init__(
            self,
            dimensionality,
            bias=True,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(512, 512, 1, padding=0, bias=bias)
        self.BN1 = nn.BatchNorm1d(512)
        self.PRelu1 = nn.PReLU(512)
        self.conv2 = nn.Conv1d(512, dimensionality, 1, padding=0, bias=bias)
        self.BN2 = nn.BatchNorm1d(dimensionality)

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
    """
    Ensemble of Teacher and DRL
    """
    def __init__(self, resnet, drl):
        super().__init__()
        self.resnet = resnet
        self.DRL = drl
        self.Teacher_mode = False

    def forward(self, x):
        """Forward prop"""
        if x.shape[1] == 6:
            if not self.Teacher_mode:
                self.Teacher_mode = True
            x = x[:, 3:, :, :]
            x_flip = FT.hflip(x)
            x = torch.cat((x, x_flip), 0)
        x = self.resnet(x)
        x = self.DRL(x)
        if self.Teacher_mode:
            x = x[:x.shape[0]//2] + x[x.shape[0]//2:]   # Flip fusion
        x = F.normalize(x, p=2, dim=1)
        return x


class Flatten(nn.Module):
    """Flattens the input"""
    def forward(self, x):
        """Forward prop"""
        return x.view(x.size(0), -1)


def l2_norm(x, axis=1):
    """l2 norm"""
    norm = torch.norm(x, 2, axis, True)
    output = torch.div(x, norm)
    return output


class SEModule(nn.Module):
    """
    SEModule
    """
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward prop"""
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class bottleneck_IR(nn.Module):
    """
    IR bottleneck module
    """
    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False), nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False), nn.BatchNorm2d(depth))

    def forward(self, x):
        """Forward prop"""
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class bottleneck_IR_SE(nn.Module):
    """
    IR bottleneck module with SE
    """
    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        """Forward prop"""
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):
    """Creates a bottleneck block."""
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1)
                                                      for i in range(num_units - 1)]


def get_blocks(num_layers):
    """Creates the block architecture for the given model."""
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


class Backbone(nn.Module):
    """
    Constructs a backbone with the given parameters.
    """
    def __init__(self, input_size, num_layers, mode='ir'):
        super().__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))
        if input_size[0] == 112:
            # Dropout is set to 0, due to the train.py structure
            self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                              nn.Dropout(p=0),
                                              Flatten(),
                                              nn.Linear(512 * 7 * 7, 512),
                                              nn.BatchNorm1d(512))
        else:
            self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                              nn.Dropout(p=0),
                                              Flatten(),
                                              nn.Linear(512 * 14 * 14, 512),
                                              nn.BatchNorm1d(512))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = nn.Sequential(*modules)

        self._initialize_weights()

    def forward(self, x):
        """Forward prop"""
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        """Initializes the weights."""
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


def create_model(input_size=(112, 112),  # pylint: disable=unused-argument
                 dimensionality=64,
                 backbone_checkpoint=None,
                 model_name="ir", model_size=152, **kwargs):
    """
    Model + DRL constructor
    """
    model = Backbone(input_size, model_size, model_name)
    if backbone_checkpoint is not None:
        try:
            model.load_state_dict(torch.load(backbone_checkpoint,
                                             map_location=torch.device('cpu')))
        except FileNotFoundError:
            print(f'Backbone checkpoint {backbone_checkpoint} not found. Please follow the '
                  'instructions in docs/FacialRecognitionSystem.md, section ## FaceID, '
                  'to download the backbone checkpoint.',
                  file=sys.stderr)
            sys.exit(1)
    for param in model.parameters():
        param.requires_grad = False
    drl = DRL(dimensionality)
    ensemble = Ensemble(model, drl)

    return ensemble


def ir_50(input_size=(112, 112),  # pylint: disable=unused-argument
          dimensionality=64,
          backbone_checkpoint=None, **kwargs):
    """
    Constructs a ir-50 model.
    """
    model = create_model(input_size, dimensionality, backbone_checkpoint, "ir", 50)

    return model


def ir_101(input_size=(112, 112),  # pylint: disable=unused-argument
           dimensionality=64,
           backbone_checkpoint=None, **kwargs):
    """
    Constructs a ir-101 model.
    """
    model = create_model(input_size, dimensionality, backbone_checkpoint, "ir", 100)

    return model


def ir_152(input_size=(112, 112),  # pylint: disable=unused-argument
           dimensionality=64,
           backbone_checkpoint=None, **kwargs):
    """
    Constructs a ir-152 model.
    """
    model = create_model(input_size, dimensionality, backbone_checkpoint, "ir", 152)

    return model


def ir_se_50(input_size=(112, 112),  # pylint: disable=unused-argument
             dimensionality=64,
             backbone_checkpoint=None, **kwargs):
    """
    Constructs a ir_se-50 model.
    """
    model = create_model(input_size, dimensionality, backbone_checkpoint, "ir_se", 50)

    return model


def ir_se_101(input_size=(112, 112),  # pylint: disable=unused-argument
              dimensionality=64,
              backbone_checkpoint=None, **kwargs):
    """
    Constructs a ir_se-101 model.
    """
    model = create_model(input_size, dimensionality, backbone_checkpoint, "ir_se", 100)

    return model


def ir_se_152(input_size=(112, 112),  # pylint: disable=unused-argument
              dimensionality=64,
              backbone_checkpoint=None, **kwargs):
    """
    Constructs a ir_se-152 model.
    """
    model = create_model(input_size, dimensionality, backbone_checkpoint, "ir_se", 152)

    return model


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
