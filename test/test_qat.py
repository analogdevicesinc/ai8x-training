#!/usr/bin/env python3
###################################################################################################
#
# Copyright (C) 2020-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Test routine for QAT
"""
import copy
import os
import sys

import torch

# Allow test to run outside of pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ai8x  # noqa: E402 pylint: disable=wrong-import-position


def create_input_data(num_channels):
    '''
    Creates random data
    '''
    inp = 2.0 * torch.rand(1, num_channels, 8, 8) - 1.0  # pylint: disable=no-member
    inp_int = torch.clamp(torch.round(128 * inp), min=-128, max=127.)  # pylint: disable=no-member
    inp = inp_int / 128.

    return inp, inp_int


def create_conv2d_layer(in_channels, out_channels, kernel_size, wide, activation):
    '''
    Creates randomly initialized layer
    '''
    ai8x.set_device(device=85, simulate=False, round_avg=False, verbose=False)
    fp_layer = ai8x.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           bias=False,
                           wide=wide,
                           activation=activation)

    fp_layer.op.weight = torch.nn.Parameter(
        (2.0 * torch.rand(out_channels,  # pylint: disable=no-member
                          in_channels,
                          kernel_size,
                          kernel_size) - 1.0)
    )
    return fp_layer


def quantize_fp_layer(fp_layer, wide, activation, num_bits):
    '''
    Creates layer with quantized leveled fp32 weights from a fp32 weighted layer
    '''
    ai8x.set_device(device=85, simulate=False, round_avg=False, verbose=False)
    in_channels = fp_layer.op.weight.shape[1]
    out_channels = fp_layer.op.weight.shape[0]
    kernel_size = fp_layer.op.weight.shape[2:]
    q_fp_layer = ai8x.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             bias=False,
                             wide=wide,
                             activation=activation,
                             weight_bits=num_bits,
                             bias_bits=8,
                             quantize_activation=True)
    q_fp_layer.op.weight = copy.deepcopy(fp_layer.op.weight)
    return q_fp_layer


def quantize_layer(q_fp_layer, wide, activation, num_bits):
    '''
    Quantizes layer
    '''
    ai8x.set_device(device=85, simulate=True, round_avg=False, verbose=False)
    in_channels = q_fp_layer.op.weight.shape[1]
    out_channels = q_fp_layer.op.weight.shape[0]
    kernel_size = q_fp_layer.op.weight.shape[2:]
    q_int_layer = ai8x.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              bias=False,
                              wide=wide,
                              activation=activation,
                              weight_bits=num_bits,
                              bias_bits=8,
                              quantize_activation=True)

    out_shift = q_fp_layer.calc_out_shift(q_fp_layer.op.weight.detach(),
                                          q_fp_layer.output_shift.detach())
    weight_scale = q_fp_layer.calc_weight_scale(out_shift)

    ai8x.set_device(device=85, simulate=False, round_avg=False, verbose=False)
    weight = q_fp_layer.clamp_weight(q_fp_layer.quantize_weight(weight_scale *
                                                                q_fp_layer.op.weight))
    q_int_weight = (2**(num_bits-1)) * weight

    q_int_layer.output_shift = torch.nn.Parameter(
        -torch.log2(weight_scale)  # pylint: disable=no-member
    )
    q_int_layer.op.weight = torch.nn.Parameter(q_int_weight)
    return q_int_layer


def test():
    '''
    Main test function
    '''
    wide_opts = [False, True]
    act_opts = [None, 'ReLU', 'Abs']
    bit_opts = [8, 4, 2, 1]

    inp, inp_int = create_input_data(512)

    for bit in bit_opts:
        for act in act_opts:
            for wide in wide_opts:
                if wide and (act is not None):
                    continue

                print(f'Testing for bits:{bit}, wide:{wide}, activation:{act}... ', end='')
                fp_layer = create_conv2d_layer(512, 16, 3, wide, act)
                q_fp_layer = quantize_fp_layer(fp_layer, wide, act, bit)
                q_int_layer = quantize_layer(q_fp_layer, wide, act, bit)

                ai8x.set_device(device=85, simulate=False, round_avg=False, verbose=False)
                q_fp_out = q_fp_layer(inp)
                ai8x.set_device(device=85, simulate=True, round_avg=False, verbose=False)
                q_int_out = q_int_layer(inp_int)

                if not wide:
                    factor = 128.
                else:
                    factor = 128. * 2.**(bit - 1)

                assert torch.isclose(q_fp_out*factor, q_int_out).all(), 'FAIL!!'
                print('PASS')

    print('\nSUCCESS!!')


if __name__ == "__main__":
    test()
