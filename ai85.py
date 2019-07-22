###################################################################################################
#
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Contains the limits of the AI85 implementation and custom PyTorch modules that take
the limits into account.
"""
WEIGHT_BITS = 8
DATA_BITS = 8
ACTIVATION_BITS = 8
FC_ACTIVATION_BITS = 16

WEIGHT_INPUTS = 256
WEIGHT_DEPTH = 768

MAX_AVG_POOL = 16

