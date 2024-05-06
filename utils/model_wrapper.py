###################################################################################################
# Copyright (C) 2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Common code for wrapped models
"""
from torch.nn.parallel import DistributedDataParallel


def unwrap(
        model,
):
    """
    Unwrap a model from torch.compile() and DistributedDataParallel
    """
    dynamo = hasattr(model, "_orig_mod")
    if dynamo:
        model = model._orig_mod  # pylint: disable=protected-access

    ddp = isinstance(model, DistributedDataParallel)
    if ddp:
        model = model.module

    return model, dynamo, ddp
