#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2019-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Check whether PyTorch supports CUDA hardware acceleration.
"""
import signal
import sys

import torch


def signal_handler(
        _signal,
        _frame,
):
    """
    Ctrl+C handler
    """
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    print("System:                ", sys.platform)
    print("Python version:        ", sys.version.replace('\n', ''))
    print("PyTorch version:       ", torch.__version__)
    print("CUDA/ROCm acceleration: ", end='')

    if not torch.cuda.is_available():
        print("NOT available in PyTorch")
    else:
        print("available in PyTorch")

    print("MPS acceleration:       ", end='')
    if not torch.backends.mps.is_available():
        print("NOT available in PyTorch")
    else:
        print("available in PyTorch")
