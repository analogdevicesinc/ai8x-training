###################################################################################################
#
# Copyright (C) 2023 Analog Devices, Inc. All Rights Reserved.
#
# Analog Devices, Inc. Default Copyright Notice:
# https://www.analog.com/en/about-adi/legal-and-risk-oversight/intellectual-property/copyright-notice.html
#
###################################################################################################


import torch
from torch import nn

"""
    Dummy Loss to use in knowledge distillation when student loss weigth is 0
"""
class DummyLoss(nn.Module):
    """
    Class for dummy loss
    """
    def __init__(self,  device='cpu'):
        """
        Initializes the loss
        """
        super().__init__()

        self.device = device
    def forward(self, output=None, target=None):
        """
        returns 0.0
        """

        return torch.tensor(0.0, device=self.device)
