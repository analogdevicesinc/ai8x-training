#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
###################################################################################################
"""
Test the CMSIS NN network generator.
"""
import numpy as np
import pytest

import cmsisnn


@pytest.mark.parametrize('test_no', [0, 1, 2, 3, 4])
def test_cmsis(test_no):
    """Main program to test cmsisnn.create_net."""

    weight = []
    bias = []
    layers = 1
    padding = [1]
    dilation = [[1, 1]]
    stride = [1]
    kernel_size = [[3, 3]]
    quantization = [8]
    pool = [0]
    pool_stride = [0]
    pool_average = [False]
    activate = [False]
    bias = [None]

    assert 0 <= test_no <= 4
    if test_no == 0:  # Passes
        chan = [2, 3]
        input_size = [2, 4, 4]

        w = np.array(
            [-16, 26, 35, -6, -40, -31, -27, -54, -51, -84, -69, -65,
             -8, -8, -13, -16, -3, 33, 48, 39, 27, 56, 50, 57, 31, 35,
             2, 8, 16, 28, 13, -18, 8, -6, 32, 20, -3, 4, 42, 41, 3, 23,
             67, 74, 8, -12, 33, 28, -25, -14, 1, 14, -3, 2], dtype=np.int64)
        w = w.reshape((chan[0] * chan[1], kernel_size[0][0], kernel_size[0][1]))
        print(w.flatten())
        weight.append(w)

        data = np.array(
            [[[85, 112, 69, 78],
              [69, 81, 51, 65],
              [45, 24, 0, 20],
              [34, 0, 15, 30]],
             [[0, 0, 3, 8],
              [0, 0, 12, 47],
              [0, 0, 0, 8],
              [0, 2, 0, 0]]],
            dtype=np.int64)
    elif test_no == 1:  # Passes
        chan = [1, 1]
        input_size = [1, 2, 2]

        w = np.array(
            [-16, 26, 35, -6, -40, -31, -27, -54, -51], dtype=np.int64)
        w = w.reshape((chan[0] * chan[1], kernel_size[0][0], kernel_size[0][1]))
        weight.append(w)

        data = np.array(
            [[[85, 112],
              [69, 81]]],
            dtype=np.int64)
    elif test_no == 2:  # Passes
        chan = [1, 3]
        input_size = [1, 2, 2]

        w = np.array(
            [-16, 26, 35, -6, -40, -31, -27, -54, -51,
             -84, -69, -65, -8, -8, -13, -16, -3, 33,
             48, 39, 27, 56, 50, 57, 31, 35, 2], dtype=np.int64)
        w = w.reshape((chan[0] * chan[1], kernel_size[0][0], kernel_size[0][1]))
        weight.append(w)

        data = np.array(
            [[[85, 112],
              [69, 81]]],
            dtype=np.int64)
    elif test_no == 3:  # Passes
        chan = [2, 1]
        input_size = [2, 2, 2]

        w = np.array(
            [-16, 26, 35, -6, -40, -31, -27, -54, -51,
             -84, -69, -65, -8, -8, -13, -16, -3, 33], dtype=np.int64)
        w = w.reshape((chan[0] * chan[1], kernel_size[0][0], kernel_size[0][1]))
        weight.append(w)

        data = np.array(
            [[[85, 112],
              [69, 81]],
             [[51, -65],
              [-45, 65]]],
            dtype=np.int64)
    elif test_no == 4:
        chan = [2, 2]
        input_size = [2, 2, 2]

        w = np.array(
            [-16, 26, 35, -6, -40, -31, -27, -54, -51,
             -84, -69, -65, -8, -8, -13, -16, -3, 33,
             48, 39, 27, 56, 50, 57, 31, 35, 2, 8,
             16, 28, 13, -18, 8, -6, 32, 20], dtype=np.int64)
        w = w.reshape((chan[0] * chan[1], kernel_size[0][0], kernel_size[0][1]))
        weight.append(w)

        data = np.array(
            [[[85, 112],
              [69, 81]],
             [[51, -65],
              [-45, 65]]],
            dtype=np.int64)

    assert data.size == chan[0]*input_size[1]*input_size[2]
    assert chan[0] == input_size[0]
    assert w.size == chan[0]*kernel_size[0][0]*kernel_size[0][1]*chan[1]

    cmsisnn.create_net('test_cmsis', True, False, True,
                       layers, input_size, kernel_size, quantization,
                       chan, padding, dilation, stride,
                       pool, pool_stride, pool_average, activate,
                       data, weight, bias, None, None,
                       'main', 'tests', 'log.txt',
                       'weights.h', 'sampledata.h',
                       False)


if __name__ == '__main__':
    for i in range(5):
        test_cmsis(i)
