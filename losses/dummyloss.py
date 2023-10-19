###################################################################################################
#
# Copyright (C) 2023 Analog Devices, Inc. All Rights Reserved.
#
# Analog Devices, Inc. Default Copyright Notice:
# https://www.analog.com/en/about-adi/legal-and-risk-oversight/intellectual-property/copyright-notice.html
#
###################################################################################################

"""
Dummy Loss to use in knowledge distillation when student loss weigth is 0
"""
import torch
from torch import nn


class DummyLoss(nn.Module):

    def __init__(self,  device='cpu'):
        super().__init__()

        self.device = device

    def forward(self, output, target):
        """
        """

        return torch.tensor(0.0, device=self.device)