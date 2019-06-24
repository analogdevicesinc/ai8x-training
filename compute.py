###################################################################################################
# Copyright (C) 2018-2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
###################################################################################################
"""
Pure Python implementation of Conv1d, Conv2d, and Linear. Allows debug of individual accumulations.
Compatible with PyTorch
"""
import numpy as np
import stats


def conv2d(data, weight, bias, input_size, out_channels, kernel_size, stride, pad,
           dilation, output, debug=False):
    """
    Compute a 2D convolution.

    SIMPLIFIED TO REMOVE GROUPS

    Note that all PyTorch numbers are ordered (C, H, W)
    """
    in_channels = input_size[0]

    # Compute 2D convolution
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
                                stats.true_macc += 1
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


def conv1d(data, weight, bias, input_size, out_channels, kernel_size, stride, pad,
           dilation, output, debug=False):
    """
    Compute a 1D convolution.

    SIMPLIFIED TO REMOVE GROUPS

    Note that all PyTorch numbers are ordered (C, L)
    """
    in_channels = input_size[0]

    # Compute 1D convolution
    for k in range(out_channels):
        out_offs = 0
        for x in range(-pad, input_size[1] - dilation * (kernel_size - 1) + pad, stride):
            val = np.int64(0)
            for c in range(in_channels):
                for w in range(kernel_size):
                    src_offs = x + w * dilation
                    if src_offs >= 0 and src_offs < input_size[1]:
                        val += weight[k][c][w] * data[c][src_offs]
                        stats.true_macc += 1
                        if debug:
                            print(f'k={k}, c={c}, x={x}, src_offs={src_offs}, '
                                  f'wt_offs={w}: weight*data={weight[k][c][w]}'
                                  f'*{data[c][src_offs]} -> accumulator = {val}')

            if bias is not None:
                val += bias[k]
                if debug:
                    print(f'+bias {bias[k]} --> output[{k}][{out_offs}] = {val}')
            output[k][out_offs] = val
            out_offs += 1


def linear(data, weight, bias, in_features, out_features, output, debug=False):
    """
    Compute a fully connected layer.
    """
    for w in range(out_features):
        val = np.int64(0)
        for n in range(in_features):
            val += data[n] * weight[w][n]
            stats.true_sw_macc += 1
            if debug:
                print(f'w={w}, n={n}, weight={weight[w][n]}, data={data[n]} '
                      f'-> accumulator = {val} ')
        if bias is not None:
            val += bias[w]
            if debug:
                print(f'+bias {bias[w]} --> output[{w}] = {val}')
        output[w] = val
