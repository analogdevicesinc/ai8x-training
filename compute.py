###################################################################################################
# Copyright (C) 2018-2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
###################################################################################################
"""
Pure Python implementation of conv2d. Allows debug of individual accumulations.
Compatible with PyTorch
"""
import numpy as np


def conv2d(data, weight, bias, input_size, out_channels, kernel_size, stride, pad,
           dilation, output, debug=False):
    """
    Compute a convolution.

    SIMPLIFIED TO REMOVE GROUPS

    Note that all PyTorch numbers are ordered (C, H, W)
    """
    in_channels = input_size[0]

    # Compute convolution
    for k in range(out_channels):
        out_offs = 0
        for y in range(-pad[0],
                       input_size[1] - dilation[0] * (kernel_size[0] - 1) + pad[0],
                       stride[0]):
            for x in range(-pad[1],
                           input_size[2] - dilation[1] * (kernel_size[1] - 1) + pad[1],
                           stride[1]):
                val = np.int64(0)
                for c in range(in_channels):
                    for h in range(kernel_size[0]):
                        for w in range(kernel_size[1]):
                            if x + w * dilation[1] >= 0 and \
                               x + w * dilation[1] < input_size[2] and \
                               y + h * dilation[0] >= 0 and \
                               y + h * dilation[0] < input_size[1]:
                                src_offs = x + w * dilation[1] + \
                                           (y + h * dilation[0]) * input_size[2]
                                wt_offs = h * kernel_size[0] + w
                                val += weight[k][c][wt_offs] * data[c][src_offs]
                                if debug:
                                    print(f'k={k}, c={c}, x={x}, y={y}, src_offs={src_offs}, '
                                          f'wt_offs={wt_offs}: weight*data={weight[k][c][wt_offs]}'
                                          f'*{data[c][src_offs]} -> accumulator = {val}')

                if bias is not None:
                    val += bias[k]
                    if debug:
                        print(f'+bias {bias[k]} --> output[{k}][{out_offs}] = {val}')
                output[k][out_offs] = val
                out_offs += 1
