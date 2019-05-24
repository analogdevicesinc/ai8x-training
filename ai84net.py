"""
Networks that fit into AI84
"""
import torch.nn as nn


__all__ = ['AI84Net5', 'ai84net5']

AI84_WEIGHT_INPUTS = 64
AI84_WEIGHT_DEPTH = 128


class AI84Net5(nn.Module):

    def __init__(self, num_classes=10, num_channels=3, dimensions=(28, 28),
                 clamp_activation_8bit=False, integer_activation=False,
                 planes=60, pool=4, fc_inputs=12, bias=False):
        super(AI84Net5, self).__init__()

        # AI84 Limits
        assert planes + num_channels <= AI84_WEIGHT_INPUTS
        assert planes + fc_inputs <= AI84_WEIGHT_DEPTH-1
        assert pool < 5
        assert dimensions[0] == dimensions[1]

        self.clamp_activation_8bit = clamp_activation_8bit
        self.integer_activation = integer_activation

        # Keep track of image dimensions so one constructor works for all image sizes
        dim = dimensions[0]

        self.conv1 = nn.Conv2d(num_channels, planes, kernel_size=3,
                               stride=1, padding=2, bias=bias)
        dim += 2  # padding -> 30x30
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1 if pool == 3 else 0)
        if pool != 3:
            dim -= 2  # stride of 2 -> 14x14, else 15x15
        dim //= 2
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1 if pool == 3 else 2, bias=bias)
        if pool != 3:
            dim += 2  # padding 2 -> 16x16, else 15x15
        self.conv3 = nn.Conv2d(planes, AI84_WEIGHT_DEPTH-planes-fc_inputs, kernel_size=3,
                               stride=1, padding=1, bias=bias)
        # no change in dimensions
        self.avgpool = nn.AvgPool2d(pool)
        dim //= pool  # pooling -> 4x4, else 3x3 or 5x5
        self.conv4 = nn.Conv2d(AI84_WEIGHT_DEPTH-planes-fc_inputs, fc_inputs, kernel_size=3,
                               stride=1, padding=1, bias=bias)
        # no change in dimensions
        self.fc = nn.Linear(fc_inputs*dim*dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def clamp_activation(self, x):
        if self.clamp_activation_8bit:
            x = x.clamp(min=-128, max=127)
        if self.integer_activation:
            x = x.round()
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.clamp_activation(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.clamp_activation(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.clamp_activation(x)
        x = self.avgpool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.clamp_activation(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ai84net5(pretrained=False, **kwargs):
    """
    Constructs a AI84Net-5 model.
    """
    assert not pretrained
    return AI84Net5(**kwargs)
