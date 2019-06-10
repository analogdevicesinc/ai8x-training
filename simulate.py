###################################################################################################
# Copyright (C) 2018-2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
###################################################################################################
"""
Simulate a single CNN layer
"""
import numpy as np
from compute import conv2d, linear
from tornadocnn import BIAS_DIV


def cnn_layer(layer, verbose,
              input_size, kernel_size, output_channels, padding, dilation, stride,
              pool, pool_stride, pool_average, do_activation,
              kernel, bias, data, bits=8, ai85=False, debug=False):
    """
    Perform pooling and convolution for one layer
    """
    if verbose:
        print(f"LAYER {layer}...\n")

        print(f"{input_size[0]}x{input_size[1]}x{input_size[2]} INPUT DATA:")
        print(data)
        print('')

    if pool[0] > 1 or pool[1] > 1:
        pooled_size = [input_size[0],
                       (input_size[1] + pool_stride[0] - pool[0]) // pool_stride[0],
                       (input_size[2] + pool_stride[1] - pool[1]) // pool_stride[1]]
        pooled = np.empty(shape=(pooled_size[0], pooled_size[1], pooled_size[2]),
                          dtype=np.int64)
        for c in range(input_size[0]):
            for row in range(0, pooled_size[1]*pool_stride[0], pool_stride[0]):
                for col in range(0, pooled_size[2]*pool_stride[1], pool_stride[1]):
                    if pool_average:
                        avg = np.average(data[c][row:row+pool[0], col:col+pool[1]])
                        if avg < 0:
                            val = np.ceil(avg).astype(np.int64).clip(min=-128, max=127)
                        else:
                            val = np.floor(avg).astype(np.int64).clip(min=-128, max=127)
                    else:
                        val = np.amax(data[c][row:row+pool[0], col:col+pool[1]])
                    pooled[c][row//pool_stride[0]][col//pool_stride[1]] = val
        if verbose:
            print(f"{pool[0]}x{pool[1]} {'AVERAGE' if pool_average else 'MAX'} "
                  f"POOLING, STRIDE {pool_stride[0]}/{pool_stride[1]} "
                  f"{input_size} -> {pooled_size}:")
            print(pooled)
            print('')
    else:
        pooled_size = input_size
        pooled = data

    if verbose:
        print(f"{kernel_size[0]}x{kernel_size[1]} KERNEL(S):")
        print(kernel)
        print(f"BIAS: {bias}\n")

    kernel = kernel.reshape((output_channels, input_size[0], -1))
    pooled = pooled.reshape((pooled_size[0], -1))

    out_size = [output_channels,
                (pooled_size[1] - dilation[0] * (kernel_size[0] - 1) - 1 +
                 2 * padding[0]) // stride[0] + 1,
                (pooled_size[2] - dilation[1] * (kernel_size[1] - 1) - 1 +
                 2 * padding[1]) // stride[1] + 1]
    out_buf = np.full(shape=(out_size[0], out_size[1]*out_size[2]),
                      fill_value=np.nan, dtype=np.int64)

    if bias is not None:
        bias = bias * BIAS_DIV

    conv2d(data=pooled,
           weight=kernel,
           bias=bias,
           input_size=pooled_size,
           out_channels=output_channels,
           kernel_size=kernel_size,
           stride=stride,
           pad=padding,
           dilation=dilation,
           output=out_buf,
           debug=debug)

    out_buf = out_buf.reshape((out_size))

    if verbose:
        print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} FULL-RES OUTPUT:")
        print(out_buf)
        print('')

    out_buf = np.floor(0.5 + out_buf / 128).astype(np.int64)
    np.clip(out_buf, -(2**bits-1), 2**(bits-1)-1, out_buf)

    if verbose:
        print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} OUTPUT "
              f"{'BEFORE ACTIVATION' if do_activation else '(NO ACTIVATION)'}:")
        print(out_buf)
        print('')

    if do_activation:
        np.clip(out_buf, 0, 2**(bits-1)-1, out_buf)

        if verbose:
            print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} ACTIVATED OUTPUT:")
            print(out_buf)
            print('')

    return out_buf, out_size


def linear_layer(verbose, do_activation,
                 weight, bias, data, bits=16, debug=False):
    """
    Perform one linear layer.
    """
    in_features = data.shape[0]
    out_features = weight.shape[0]

    if verbose:
        print("CLASSIFICATION LAYER...\n")
        print(f"INPUT DATA (size {in_features}):")
        print(data)
        print('')

        print(f"WEIGHTS (size {in_features * out_features}):")
        print(weight)
        print(f"BIAS: {bias}\n")

    out_buf = np.empty(out_features, dtype=np.int64)
    linear(data=data, weight=weight, bias=bias,
           in_features=in_features, out_features=out_features,
           output=out_buf, debug=debug)
    out_buf = np.floor(0.5 + out_buf / 128).astype(np.int64)
    np.clip(out_buf, -2**(bits-1), 2**(bits-1)-1, out_buf)

    if verbose:
        print(f"OUTPUT (size {out_features}):")
        print(out_buf)
        print('')

    if do_activation:
        np.clip(out_buf, 0, 2**(bits-1)-1, out_buf)

        if verbose:
            print(f"ACTIVATED OUTPUT (size {out_features}):")
            print(out_buf)
            print('')

    return out_buf, out_features
