#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2018-2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
###################################################################################################
"""
Simulation test generator program for Tornado CNN
"""
import argparse
import os
import signal
import sys

import numpy as np
import tabulate
import torch
import yaml

import sampledata
from utils import argmin, ffs, fls, popcount


# AI84
APB_BASE = 0x50100000

# CNN hardware parameters
C_MAX_LAYERS = 16
P_TRAMABITS = 8
P_BRAMABITS = 8
P_MASKABITS = 7
P_LYRBITS = 5
C_CNN = 4
C_CNN_BASE = 0
C_TRAM_BASE = C_CNN_BASE + 0x800
C_MRAM_BASE = C_CNN_BASE + 0x4800
C_BRAM_BASE = C_CNN_BASE + 0xC800
C_SRAM_BASE = C_CNN_BASE + 0x10000
C_TILE_OFFS = 0x100000
P_NUMTILES = 4
P_NUMPRO = 16  # Processors per tile
P_SHARED = 4  # Processors sharing a data memory

INSTANCE_SIZE = 1024  # x32
TILE_SIZE = 0x40000
MEM_SIZE = INSTANCE_SIZE*P_NUMPRO*P_NUMTILES//P_SHARED  # x32
MAX_DATA_DIM = 512  # Max theoretical size of single channel, used for command line validation only
MAX_CHANNELS = P_NUMPRO*P_NUMTILES

AI85 = False


def s2u(i):
    """
    Convert signed 8-bit int to unsigned
    """
    if i < 0:
        i += 256
    return i


def u2s(i):
    """
    Convert unsigned 8-bit int to signed
    """
    if i > 127:
        i -= 256
    return i


def conv2d(data, weight, bias, input_size, out_channels, kernel_size, stride, pad,
           dilation, output, debug=False):
    """
    Compute a convolution, and then run same data through PyTorch

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
                    if AI85:
                        val += bias[k] * 128
                    else:
                        val += bias[k]
                    if debug:
                        print(f'+bias {bias[k]}{"*128" if AI85 else ""} --> '
                              f'output[{k}][{out_offs}] = {val}')
                output[k][out_offs] = val
                out_offs += 1


def cnn_layer(layer, verbose,
              input_size, kernel_size, output_channels, padding, dilation, stride,
              pool, pool_stride, pool_average, do_activation,
              kernel, bias, data, debug=False):
    """
    Perform pooling and convolution
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
    np.clip(out_buf, -128, 127, out_buf)

    if verbose:
        print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} OUTPUT "
              f"{'BEFORE ACTIVATION' if do_activation else '(NO ACTIVATION)'}:")
        print(out_buf)
        print('')

    if do_activation:
        np.clip(out_buf, 0, 127, out_buf)

        if verbose:
            print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} ACTIVATED OUTPUT:")
            print(out_buf)
            print('')

    return out_buf, out_size


def create_sim(prefix, verbose, debug, debug_computation, no_error_stop, overwrite_ok, log,
               apb_base, layers, processor_map,
               input_size, kernel_size, chan, padding, dilation, stride,
               pool, pool_stride, pool_average, activate,
               data, kernel, bias, big_data, output_map, split,
               in_offset, out_offset,
               input_filename, output_filename, c_filename,
               base_directory, runtest_filename, log_filename,
               zero_unused, timeout, block_mode, verify_writes,
               c_library=False):
    """
    Chain multiple CNN layers, create and save input and output
    """

    # Remove extraneous input layer configurations (when --stop-after is used)
    if len(processor_map) > layers:
        output_map = processor_map[layers]
        processor_map = processor_map[:layers]
    chan = chan[:layers+1]
    out_offset = out_offset[:layers]

    # Trace output sizes of the network and fix up all pool_stride values
    dim = [[input_size[1], input_size[2]]]
    for ll in range(layers):
        if pool[ll] > 0:
            pooled_size = [(dim[ll][0] + pool_stride[ll] - pool[ll]) // pool_stride[ll],
                           (dim[ll][1] + pool_stride[ll] - pool[ll]) // pool_stride[ll]]
        else:
            pool_stride[ll] = 1
            pooled_size = dim[ll]
        dim.append([(pooled_size[0] - dilation[0] * (kernel_size[0] - 1) - 1 +
                     2 * padding[ll]) // stride[ll] + 1,
                    (pooled_size[1] - dilation[1] * (kernel_size[1] - 1) - 1 +
                     2 * padding[ll]) // stride[ll] + 1])

    # Complete list of offsets
    out_offset.insert(0, in_offset)  # All input locations
    processor_map.append(output_map)  # Final map

    # Write memfile for input

    # Create comment of the form "k1_b0-1x32x32b_2x2s2p14-..."
    test_name = prefix
    for ll in range(layers):
        test_name += f'-{chan[ll]}' \
                     f'x{dim[ll][0]}x{dim[ll][1]}' \
                     f'{"b" if big_data[ll] else "l"}_' \
                     f'{"avg" if pool_average[ll] and pool[ll] > 0 else ""}' \
                     f'{"max" if not pool_average[ll] and pool[ll] > 0 else ""}' \
                     f'{pool[ll]}x{pool[ll]}s{pool_stride[ll]}' \
                     f'p{padding[ll]}' \
                     f'm{chan[ll+1]}' \
                     f'{"_relu" if activate[ll] else ""}'
    print(f'{test_name}...')

    os.makedirs(os.path.join(base_directory, test_name), exist_ok=True)

    def apb_write(addr, val, comment='', check=False, no_verify=False, rv=False):
        """
        Write address `addr` and data `val` to .mem or .c file.
        If `rv` or `check`, then only verify.
        """
        assert val >= 0
        assert addr >= 0
        addr += apb_base
        if block_mode:
            memfile.write(f'@{apb_write.foffs:04x} {addr:08x}\n')
            memfile.write(f'@{apb_write.foffs+1:04x} {val:08x}\n')
        else:
            if rv:
                memfile.write(f'  if (*((volatile uint32_t *) 0x{addr:08x}) != 0x{val:08x}) '
                              f'return 0;{comment}\n')
            elif check:
                memfile.write(f'  if (*((volatile uint32_t *) 0x{addr:08x}) != 0x{val:08x}) '
                              f'return 0;{comment}\n')
            else:
                memfile.write(f'  *((volatile uint32_t *) 0x{addr:08x}) = 0x{val:08x};{comment}\n')
                if verify_writes and not no_verify:
                    memfile.write(f'  if (*((volatile uint32_t *) 0x{addr:08x}) != 0x{val:08x}) '
                                  'return 0;\n')
        apb_write.foffs += 2

    # Redirect stdout?
    if log:
        sys.stdout = open(os.path.join(base_directory, test_name, log_filename), 'w')
        print(f'{test_name}')

    if block_mode:
        filename = input_filename + '.mem'
    else:
        filename = c_filename + '.c'
    with open(os.path.join(base_directory, test_name, filename), mode='w') as memfile:

        memfile.write(f'// {test_name}\n')
        memfile.write(f'// Created using {" ".join(str(x) for x in sys.argv)}\n')

        # Human readable description of test
        memfile.write(f'// Configuring input for {layers} layer{"s" if layers > 1 else ""}\n')

        for ll in range(layers):
            memfile.write(f'// Layer {ll+1}: {chan[ll]}x{dim[ll][0]}x{dim[ll][1]} '
                          f'{"(CHW/big data)" if big_data[ll] else "(HWC/little data)"}, ')
            if pool[ll] > 0:
                memfile.write(f'{pool[ll]}x{pool[ll]} {"avg" if pool_average[ll] else "max"} '
                              f'pool with stride {pool_stride[ll]}')
            else:
                memfile.write(f'no pooling')
            memfile.write(f', {kernel_size[0]}x{kernel_size[1]} convolution '
                          f'with stride {stride[ll]} '
                          f'pad {padding[ll]}, {chan[ll+1]}x{dim[ll+1][0]}x{dim[ll+1][1]} out\n')

        memfile.write('\n')

        if not block_mode:
            memfile.write('#include "global_functions.h"\n')
            memfile.write('#include <stdlib.h>\n')
            memfile.write('#include <stdint.h>\n')
            if c_library:
                memfile.write('#include <string.h>\n')
            memfile.write('\nint cnn_check(void);\n\n')
            memfile.write('void cnn_wait(void)\n{\n')
            memfile.write(f'  while ((*((volatile uint32_t *) 0x{apb_base + C_CNN_BASE:08x}) '
                          '& (1<<12)) != 1<<12) ;\n}\n\n')
            memfile.write('int cnn_load(void)\n{\n')

        def apb_write_ctl(tile, reg, val, comment=None):
            """
            Write global control address and data to .mem file
            """
            if comment is None:
                comment = f' // global ctl {reg}'
            addr = C_TILE_OFFS*tile + C_CNN_BASE + reg*4
            apb_write(addr, val, comment)

        def apb_write_reg(tile, layer, reg, val, debug=False, comment=None):
            """
            Write register address and data to .mem file
            """
            if comment is None:
                comment = f' // reg {reg}'
            addr = C_TILE_OFFS*tile + C_CNN_BASE + C_CNN*4 + reg*4 * 2**P_LYRBITS + layer*4
            apb_write(addr, val, comment)
            if debug:
                print(f'T{tile} L{layer} R{reg:02} ({addr:08x}): {val:08x}')

        def apb_write_kern(ll, ch, idx, k):
            """
            Write kernel to .mem file
            """
            assert ch < MAX_CHANNELS
            assert idx < 2**P_MASKABITS
            addr = C_TILE_OFFS*(ch // P_NUMPRO) + C_MRAM_BASE + (ch % P_NUMPRO) * \
                2**P_MASKABITS * 16 + idx * 16
            apb_write(addr, k[0] & 0xff, no_verify=True,
                      comment=f' // layer {ll}: processor {ch} kernel #{idx}')
            apb_write(addr+4, (k[1] & 0xff) << 24 | (k[2] & 0xff) << 16 |
                      (k[3] & 0xff) << 8 | k[4] & 0xff, no_verify=True)
            apb_write(addr+8, (k[5] & 0xff) << 24 | (k[6] & 0xff) << 16 |
                      (k[7] & 0xff) << 8 | k[8] & 0xff, no_verify=True)
            apb_write(addr+12, 0, no_verify=True)  # Execute write
            if verify_writes:
                apb_write(addr, k[0] & 0xff, check=True)
                apb_write(addr+4, (k[1] & 0xff) << 24 | (k[2] & 0xff) << 16 |
                          (k[3] & 0xff) << 8 | k[4] & 0xff, check=True)
                apb_write(addr+8, (k[5] & 0xff) << 24 | (k[6] & 0xff) << 16 |
                          (k[7] & 0xff) << 8 | k[8] & 0xff, check=True)
                apb_write(addr+12, 0, check=True)  # Execute write

        def apb_write_bias(tile, offs, bias):
            """
            Write bias value to .mem file
            """
            addr = C_TILE_OFFS*tile + C_BRAM_BASE + offs * 4
            apb_write(addr, bias & 0xff, f' // bias')

        def apb_write_tram(tile, proc, offs, d, comment=''):
            """
            Write TRAM value to .mem file
            """
            addr = C_TILE_OFFS*tile + C_TRAM_BASE + proc * 2**P_TRAMABITS * 4 + offs * 4
            apb_write(addr, d, f' // {comment}TRAM T {tile} P {proc} #{offs}')

        apb_write.foffs = 0

        # Calculate the tiles needed, and tiles and processors used overall
        processors_used = 0
        tile_map = []
        for ll in range(layers):
            bits = processor_map[ll]
            processors_used |= bits

            if popcount(processor_map[ll]) != chan[ll]:
                print(f'Layer {ll} has {chan[ll]} inputs, but enabled '
                      f'processors {processor_map[ll]:016x} does not match.')
                sys.exit(1)
            if chan[ll] > MAX_CHANNELS:
                print(f'Layer {ll} is configured for {chan[ll]} inputs, which exceeds '
                      f'the system maximum of {MAX_CHANNELS}.')
                sys.exit(1)
            this_map = []
            for tile in range(P_NUMTILES):
                if (processor_map[ll] >> tile*P_NUMPRO) & (2**P_NUMPRO-1) != 0:
                    this_map.append(tile)
            tile_map.append(this_map)

        tiles_used = []
        for tile in range(P_NUMTILES):
            if ((processors_used | processor_map[layers]) >> tile*P_NUMPRO) & (2**P_NUMPRO-1) != 0:
                tiles_used.append(tile)

        # Initialize CNN registers

        # Disable completely unused tiles
        for tile in range(P_NUMTILES):
            if tile not in tiles_used:
                apb_write_ctl(tile, 0, 0, comment=f' // disable tile {tile}')

        memfile.write('\n')

        # Configure global control registers for used tiles
        for _, tile in enumerate(tiles_used):
            # Zero out Tornado RAM ('zero_tram')
            if c_library:
                addr = apb_base + C_TILE_OFFS*tile + C_TRAM_BASE
                memfile.write(f'  memset((uint32_t *) 0x{addr:08x}, 0, '
                              f'{2**P_TRAMABITS * P_NUMPRO * 4}); // zero TRAM tile {tile}\n')
            else:
                for p in range(P_NUMPRO):
                    for offs in range(2**P_TRAMABITS):
                        apb_write_tram(tile, p, offs, 0, comment='zero ')

            memfile.write('\n')

            # Stop state machine - will be overwritten later
            apb_write_ctl(tile, 0, 0x06, comment=' // stop SM')  # ctl reg
            # SRAM Control - does not need to be changed
            apb_write_ctl(tile, 1, 0x40e, comment=' // SRAM ctl')  # sram reg
            # Number of layers
            apb_write_ctl(tile, 2, layers-1, comment=' // layer count')  # layer max count reg
            memfile.write('\n')

        # Kernels ('load_mask')
        # Stack kernels; write only the kernels needed
        chan_kern_max = [0] * MAX_CHANNELS
        kern_offs = [0] * layers
        kern_len = [0] * layers
        kernel_map = [[None] * 2**P_MASKABITS for i in range(MAX_CHANNELS)]
        for ll in range(layers):
            first_channel = ffs(processor_map[ll])
            last_channel = fls(processor_map[ll])
            ch = 0
            for c in range(first_channel, last_channel+1):
                if (processor_map[ll] >> c) & 1 == 0:
                    # Unused processor
                    continue
                # Get highest offset for all used channels
                kern_offs[ll] = max(chan_kern_max[c], kern_offs[ll])

            # Determine the number of kernels that need to be programmed. Since each instance
            # spans 4 processors, kernels for all instances that have a single processor enabled
            # need to be written, i.e. round down the first and round up the last
            # FIXME: Deal with gaps that are full instances (i.e., all bits off for an instance)
            next_layer_map = processor_map[ll+1]
            kern_len[ll] = ((fls(next_layer_map) + P_SHARED-1) & ~(P_SHARED-1)) \
                - ((ffs(next_layer_map)) & ~(P_SHARED-1))

            if kern_offs[ll] + kern_len[ll] > 2**P_MASKABITS:
                print(f'Kernel memory exceeded at layer {ll}; offset: {kern_offs[ll]}, '
                      f'needed: {kern_len[ll]}')
                print('\nKernel map so far:')
                table = tabulate.tabulate(kernel_map, tablefmt='plain', missingval='X')
                print('-' * (2**P_MASKABITS * 2 - 1))
                print(table.replace('  ', ' '))
                print('-' * (2**P_MASKABITS * 2 - 1))

            for c in range(first_channel, last_channel+1):
                if (processor_map[ll] >> c) & 1 == 0:
                    # Unused processor
                    continue

                # print(f'c {c}, Kernel dimensions layer {ll}: {kernel[ll].shape}')
                # print(f'should be {chan[ll]*chan[ll+1]}')
                # Start at the first used instance
                this_map = next_layer_map >> (ffs(next_layer_map) & ~(P_SHARED-1))
                offs = 0
                for i in range(chan[ll+1]):
                    while this_map & 1 == 0:
                        assert this_map != 0
                        # FIXME: Deal with full-instance gaps, see above
                        offs += 1
                        this_map >>= 1
                    this_map >>= 1

                    # print(f'i {i} L {ll} C {c}: Channel index {i + ch*chan[ll+1]} -> ', end='')
                    k = kernel[ll][ch + i*chan[ll]].flatten()
                    if debug:
                        print(f'Channel {c} Layer {ll} m{i}/{chan[ll+1]-1}: {k}')
                    apb_write_kern(ll, c, chan_kern_max[c] + i + offs, k)

                    # Update kernel map
                    kernel_map[c][chan_kern_max[c] + i + offs] = ll

                assert kern_len[ll] == offs + i + 1
                chan_kern_max[c] = kern_offs[ll] + kern_len[ll]
                ch += 1

        if verbose:
            print('\nKernel Map:')
            table = tabulate.tabulate(kernel_map, tablefmt='plain', missingval='X')
            print('-' * (2**P_MASKABITS * 2 - 1))
            print(table.replace('  ', ' '))
            print('-' * (2**P_MASKABITS * 2 - 1))

        # Bias ('zero_bias_ram')
        # Each tile has one bias memory (size 2**P_BRAMABITS bytes). Use only the bias memory in
        # one selected tile for the layer, and only if the layer uses a bias. Keep track of the
        # offsets so they can be programmed into the mask count register later.
        tile_bias_max = [0] * P_NUMTILES
        bias_offs = [None] * layers
        bias_tile = [None] * layers
        for ll in range(layers):
            if bias[ll] is None:
                continue
            if len(bias[ll]) != chan[ll+1]:
                print(f'Layer {ll}: output channel count {chan[ll+1]} does not match the number '
                      f'of bias values {len(bias[ll])}')
                sys.exit(1)

            # Pick the tile with the least amount of data in it
            tile = argmin(tile_bias_max[t] for t in tile_map[ll])
            if tile_bias_max[tile] + chan[ll+1] > 2**P_BRAMABITS:
                print(f'Layer {ll}: bias memory capacity exceeded - available tiles: '
                      f'{tile_map[ll]}, used so far: {tile_bias_max}, needed: {chan[ll+1]}')
                sys.exit(1)

            bias_tile[ll] = tile
            bias_offs[ll] = tile_bias_max[tile]
            # Each layer has output_channel number of bias values
            for i in range(chan[ll+1]):
                apb_write_bias(tile, bias_offs[ll] + i, bias[ll][i])
            tile_bias_max[tile] += chan[ll+1]

        if verbose:
            print('\nGlobal configuration:')
            print('---------------------')
            print(f'Used processors    = {processors_used:016x}')
            print(f'Used tiles         = {tiles_used}')
            print('\nPer-tile configuration:')
            print('-----------------------')
            print(f'Used bias memory   = {tile_bias_max}')
            print('\nPer-layer configuration:')
            print('------------------------')
            print(f'Number of channels = {chan[:layers]} -> {chan[layers]} outputs')
            print('Processor map      = [',
                  ', '.join('{:016x}'.format(k) for k in processor_map[:layers]), ']',
                  f' -> {processor_map[layers]:016x} output', sep='',)
            print(f'Tile map           = {tile_map}')
            print(f'Kernel offsets     = {kern_offs}')
            print(f'Tile with bias     = {bias_tile}')
            print(f'Bias offsets       = {bias_offs}')
            print('')

        def apb_write_byte_flush(offs, comment=''):
            if apb_write_byte.num > 0:
                woffs = apb_write_byte.data_offs - apb_write_byte.num
                if apb_write_byte_flush.mem[woffs >> 2]:
                    print(f'Overwriting location {woffs:08x}')
                    if not no_error_stop:
                        sys.exit(1)
                apb_write(woffs, apb_write_byte.data, comment)
                apb_write_byte_flush.mem[woffs >> 2] = True
                apb_write_byte.num = 0
                apb_write_byte.data = 0
            apb_write_byte.data_offs = offs

        apb_write_byte_flush.mem = [False] * C_TILE_OFFS * P_NUMTILES

        def apb_write_byte(offs, val, comment=''):
            """
            Collect bytes and write them to word memory.
            If discontiguous, flush with zero padding.
            """
            if offs != apb_write_byte.data_offs:
                apb_write_byte_flush(offs)

            # Collect and write if multiple of 4 (little endian byte order)
            apb_write_byte.data |= (val & 0xff) << (8*apb_write_byte.num)
            apb_write_byte.num += 1
            apb_write_byte.data_offs += 1
            if apb_write_byte.num == 4:
                apb_write_byte_flush(offs+1, comment)

        apb_write_byte.data = 0
        apb_write_byte.num = 0
        apb_write_byte.data_offs = 0

        # Configure per-layer control registers
        for _, tile in enumerate(tiles_used):
            for ll in range(layers):
                memfile.write(f'\n  // Tile {tile} layer {ll}\n')

                # Configure row count ('config_cnn_rcnt')
                # [7:0] maxcount: lower 8 bits = total of width + pad - 1
                # [9:8] pad: 2 bits pad
                apb_write_reg(tile, ll, 0, (padding[ll] << 8) | (dim[ll][0]-1 + 2*padding[ll]),
                              comment=' // rows')

                # Configure column count ('config_cnn_ccnt')
                # [7:0] width including padding - 1
                # [9:8] pad count (0 = no pad, 1 = half pad, 2 = full pad)
                apb_write_reg(tile, ll, 1, padding[ll] << 8 | (dim[ll][1]-1 + 2 * padding[ll]),
                              comment=' // columns')

                # Configure pooling row count ('config_cnn_prcnt')
                apb_write_reg(tile, ll, 3, max(1, pool[ll]-1), comment=' // pooling rows')

                # Configure pooling column count ('config_cnn_pccnt')
                apb_write_reg(tile, ll, 4, max(1, pool[ll]-1), comment=' // pooling columns')

                # Configure pooling stride count ('config_cnn_stride')
                apb_write_reg(tile, ll, 5, pool_stride[ll]-1, comment=' // pooling stride')

                # Configure SRAM write pointer ('config_cnn_wptr') -- write ptr is global
                # Get offset to first available instance of the first used processor of the next
                # layer.
                offs = (ffs(processor_map[ll+1]) & ~(P_SHARED-1)) * INSTANCE_SIZE
                apb_write_reg(tile, ll, 6, out_offset[ll+1] // 4 + offs, debug,
                              comment=' // SRAM write ptr')

                # Configure write pointer mask offset count ('config_cnn_woff')
                # [15:0]  Timeslot offset
                #         [11:0]  12 bits for memory - word address every time we reach mask limit
                #         [13:12] instance in group
                #         [15:14] by-16 group
                # [31:16] Mask offset (0x10000000, required when writing more than 4 masks)
                if chan[ll] * kern_len[ll] > 4:
                    val = 0x10000000
                else:
                    val = 0
                apb_write_reg(tile, ll, 7, val, debug, comment=' // mask offset count')

                # Configure sram read ptr count ('config_cnn_rptr') -- read ptr is local
                # Source address must match write pointer of previous layer (minus global offset)
                apb_write_reg(tile, ll, 8, out_offset[ll] // 4, comment=' // SRAM read ptr')

                # Configure per-layer control
                # [3:0] s_slave: enable the by-4 group within the by-16 mask RAM to slave
                #                to first input volume; also enable timeslot
                # [4]   m_slave: slaves to 16x masters
                # [5]   master: sums all 16 processor outputs (vs 4 sums)
                # [6]   parallel: equals CHW/big data (per layer control)
                # [7]   pool_enable
                # [8]   maxpool_enable
                # [9]   activation_enable
                # [10]  cpad_only (column pad only, no row pad) for parallel processing
                # [11]  sramlsrc: global/local SRAM data input select
                # [15:12] cnnsiena: enable externally sourced summed values from other processors
                val = (0x200 if activate[ll] else 0) | \
                      (0x100 if not pool_average[ll] else 0) | \
                      (0x80 if pool[ll] > 1 else 0) | \
                      (0x40 if big_data[ll] else 0) | \
                      (0x820)
                if tile == tile_map[ll][0]:
                    # Set external source for other active processing tiles (can be zero if no
                    # other tiles are processing). Do not set the bit corresponding to this tile
                    # (e.g., if tile == 0, do not set bit 12)
                    sources = 0
                    for t in range(tile_map[ll][0]+1, P_NUMTILES):
                        # See if any processors other than this one are operating
                        # and set the cnnsiena bit if true
                        if processor_map[ll] >> (t * P_NUMPRO) & (2**P_NUMPRO-1) != 0:
                            sources |= 1 << t
                    val |= sources << 12
                apb_write_reg(tile, ll, 9, val, debug, comment=' // layer control')

                # Configure mask count ('config_cnn_mask')
                # Restriction: Every one of the mask memories will have to start from same offset
                # [6:0]   Max count (output channels)
                # [7]     RFU
                # [14:8]  Starting address for group of 16
                # [15]    RFU
                # [23:16] Bias pointer starting address
                # [24]    Bias enable
                # [31:25] RFU
                val = kern_offs[ll] << 8 | kern_len[ll]-1
                if tile == bias_tile[ll]:
                    # Enable bias only for one tile
                    val |= 0x1000000 | bias_offs[ll] << 16
                apb_write_reg(tile, ll, 10, val, debug, comment=' // mask count')

                # Configure tram pointer max ('config_cnn_tptr')
                if pool[ll] > 0:
                    val = max(0, (dim[ll][1] + pool_stride[ll] - pool[ll]) // pool_stride[ll] +
                              2*padding[ll] - 3)
                else:
                    val = max(0, dim[ll][1] + 2*padding[ll] - 3)
                apb_write_reg(tile, ll, 11, val, comment=' // TRAM ptr max')

                # Configure mask and processor enables ('config_cnn_mena')
                # [15:0]  mask enable
                # [31:16] processor enable (or the reverse?)
                # When the input data is sourced from 16 independent byte streams, all 16
                # processors and compute elements need to be enabled.  If there were only 4 input
                # channels, 0x000f000f would be correct.
                #
                # Enable at most 16 processors and masks
                bits = (processor_map[ll] >> tile*P_NUMPRO) & (2**P_NUMPRO - 1)
                apb_write_reg(tile, ll, 12, bits << 16 | bits, debug,
                              comment=' // mask and processor enables')

            if zero_unused:
                for ll in range(layers, C_MAX_LAYERS):
                    apb_write_reg(tile, ll, 0, 0, comment=f' // zero unused layer {ll} registers')
                    apb_write_reg(tile, ll, 1, 0, comment=f' // zero unused layer {ll} registers')
                    apb_write_reg(tile, ll, 3, 0, comment=f' // zero unused layer {ll} registers')
                    apb_write_reg(tile, ll, 4, 0, comment=f' // zero unused layer {ll} registers')
                    apb_write_reg(tile, ll, 5, 0, comment=f' // zero unused layer {ll} registers')
                    apb_write_reg(tile, ll, 6, 0, comment=f' // zero unused layer {ll} registers')
                    apb_write_reg(tile, ll, 7, 0, comment=f' // zero unused layer {ll} registers')
                    apb_write_reg(tile, ll, 8, 0, comment=f' // zero unused layer {ll} registers')
                    apb_write_reg(tile, ll, 9, 0, comment=f' // zero unused layer {ll} registers')
                    apb_write_reg(tile, ll, 10, 0, comment=f' // zero unused layer {ll} registers')
                    apb_write_reg(tile, ll, 11, 0, comment=f' // zero unused layer {ll} registers')
                    apb_write_reg(tile, ll, 12, 0, comment=f' // zero unused layer {ll} registers')

        # Load data memory ('admod_sram'/'lildat_sram')
        # Start loading at the first used tile
        memfile.write(f'\n\n  // {chan[0]}-channel data input\n')
        c = 0
        data_offs = 0
        step = 1 if big_data[0] else 4
        for ch in range(0, MAX_CHANNELS, step):
            if (processor_map[0] >> ch) & (2**step-1) == 0:
                # Channel or block of four channels not used for input
                continue

            # Load channel into shared memory
            tile = ch // P_NUMPRO
            group = (ch % P_NUMPRO) // P_SHARED
            new_data_offs = C_TILE_OFFS*tile + C_SRAM_BASE + INSTANCE_SIZE*4*group
            if new_data_offs == data_offs:
                print('Layer 0 processor map is misconfigured for data input. '
                      f'There is data overlap between processors {ch-1} and {ch}')
                sys.exit(1)
            data_offs = new_data_offs

            if debug:
                print(f'T{tile} L0 data_offs:      {data_offs:08x}')

            if big_data[0]:
                # CHW ("Big Data") - Separate channel sequences (BBBBB....GGGGG....RRRRR....)
                memfile.write(f'  // CHW (big data): {dim[0][0]}x{dim[0][1]}, channel {c}\n')

                chunk = input_size[1] // split

                # (Note: We do not need to flush here, since that is done at the
                # end of each channel's output below)
                if split > 1:
                    # Add top pad
                    for _ in range(padding[0]):
                        for _ in range(input_size[2]):
                            apb_write_byte(data_offs, 0)
                            data_offs += 1
                row = 0
                for s in range(split):
                    if split > 1 and s + 1 < split:
                        overlap = padding[0]
                    else:
                        overlap = 0
                    while row < (s + 1) * chunk + overlap:
                        for col in range(input_size[2]):
                            apb_write_byte(data_offs, s2u(data[c][row][col]))
                            data_offs += 1
                        row += 1
                    row -= 2*overlap  # Rewind
                    # Switch to next memory instance
                    if split > 1 and s + 1 < split:
                        new_data_offs = ((data_offs + INSTANCE_SIZE - 1) //
                                         INSTANCE_SIZE) * INSTANCE_SIZE
                        if new_data_offs != data_offs:
                            apb_write_byte_flush(0)
                        data_offs = new_data_offs
                if split > 1:
                    # Add bottom pad
                    for _ in range(padding[0]):
                        for _ in range(input_size[2]):
                            apb_write_byte(data_offs, 0)
                            data_offs += 1
                c += 1
            else:
                # HWC ("Little Data") - Four channels packed into a word (0BGR0BGR0BGR0BGR0BGR....)
                memfile.write(f'  // HWC (little data): {dim[0][0]}x{dim[0][1]}, '
                              f'channels {c} to {min(c+3, chan[0]-1)}\n')

                for row in range(input_size[1]):
                    for col in range(input_size[2]):
                        if c < chan[0]:
                            apb_write_byte(data_offs, s2u(data[c][row][col]))
                        else:
                            apb_write_byte(data_offs, 0)
                        data_offs += 1
                        # Always write multiple of four bytes even for last input
                        if c+1 < chan[0]:
                            apb_write_byte(data_offs, s2u(data[c+1][row][col]))
                        else:
                            apb_write_byte(data_offs, 0)
                        data_offs += 1
                        if c+2 < chan[0]:
                            apb_write_byte(data_offs, s2u(data[c+2][row][col]))
                        else:
                            apb_write_byte(data_offs, 0)
                        data_offs += 1
                        if c+3 < chan[0]:
                            apb_write_byte(data_offs, s2u(data[c+3][row][col]))
                        else:
                            apb_write_byte(data_offs, 0)
                        data_offs += 1
                c += 4

            apb_write_byte_flush(0)
            if c >= chan[0]:
                # Consumed all available channels
                break

        memfile.write(f'  // End of data input\n\n')

        # Enable all needed tiles except the first one
        for _, tile in enumerate(tiles_used[1:]):
            # [0] enable
            # [8] one-shot (stop after single layer)
            # cnn_ena_i <= #C_TPD pwdata[0];    # Enable
            # rdy_sel   <= #C_TPD pwdata[2:1];  # Wait states - set to max
            # ext_sync  <= #C_TPD pwdata[3];    # Slave to other group
            # calcmax_i <= #C_TPD pwdata[4];    # RFU
            # poolena_i <= #C_TPD pwdata[5];    # RFU
            # bigdata_i <= #C_TPD pwdata[6];    # RFU
            # actena_i  <= #C_TPD pwdata[7];    # RFU
            # oneshot   <= #C_TPD pwdata[8];    # One-shot
            # ext_sync  <= #C_TPD pwdata[11:9]; # See above
            apb_write_ctl(tile, 0, 0x807, comment=f' // enable tile {tile}')

        # Master control - go
        apb_write_ctl(tiles_used[0], 0, 0x07, comment=f' // master enable tile {tiles_used[0]}')

        if not block_mode:
            memfile.write('  return 1;\n}\n\nint main(void)\n{\n  icache_enable();\n')
            memfile.write('  MXC_GCR->perckcn1 &= ~0x20; // Enable AI clock\n')
            memfile.write('  if (!cnn_load()) { fail(); pass(); return 0; }\n  cnn_wait();\n')
            memfile.write('  if (!cnn_check()) fail();\n')
            memfile.write('  pass();\n  return 0;\n}\n\n')

        # End of input

    in_map = apb_write_byte_flush.mem

    if verbose:
        print('')

    # Compute layer-by-layer output and chain results into input
    for ll in range(layers):
        out_buf, out_size = cnn_layer(ll + 1, verbose,
                                      input_size, kernel_size, chan[ll+1],
                                      [padding[ll], padding[ll]], dilation,
                                      [stride[ll], stride[ll]],
                                      [pool[ll], pool[ll]],
                                      [pool_stride[ll], pool_stride[ll]],
                                      pool_average[ll],
                                      activate[ll],
                                      kernel[ll].reshape(chan[ll+1], input_size[0],
                                                         kernel_size[0], kernel_size[1]),
                                      bias[ll],
                                      data, debug=debug_computation)

        # Write .mem file for output or create the C cnn_check() function to verify the output
        out_map = [False] * C_TILE_OFFS * P_NUMTILES
        apb_write.foffs = 0  # Position in output file
        if block_mode:
            if ll == layers-1:
                filename = output_filename + '.mem'  # Final output
            else:
                filename = f'{output_filename}-{ll+1}.mem'  # Intermediate output
            filemode = 'w'
        else:
            if ll == layers-1:
                filename = c_filename + '.c'  # Final output
            else:
                filename = '/dev/null'  # Intermediate output
            filemode = 'a'
        with open(os.path.join(base_directory, test_name, filename), mode=filemode) as memfile:
            memfile.write(f'// {test_name}\n// Expected output of layer {ll+1}\n')
            if not block_mode:
                memfile.write('int cnn_check(void)\n{\n  int rv = 1;\n')

            next_layer_map = processor_map[ll+1]

            for row in range(out_size[1]):
                for col in range(out_size[2]):
                    this_map = next_layer_map
                    noffs = 0
                    for c in range(0, chan[ll+1], 4):
                        while this_map & 1 == 0:
                            assert this_map != 0
                            # FIXME: Deal with full-instance gaps, see above
                            noffs += 1
                            this_map >>= 1
                        this_map >>= 1

                        val = out_buf[c][row][col] & 0xff
                        if c+1 < chan[ll+1]:
                            val |= (out_buf[c+1][row][col] & 0xff) << 8
                        if c+2 < chan[ll+1]:
                            val |= (out_buf[c+2][row][col] & 0xff) << 16
                        if c+3 < chan[ll+1]:
                            val |= (out_buf[c+3][row][col] & 0xff) << 24

                        offs = out_offset[ll+1] + (((c + noffs) % P_NUMPRO)*INSTANCE_SIZE +
                                                   ((c + noffs) // P_NUMPRO)*TILE_SIZE +
                                                   row*out_size[2] + col)*4 + C_SRAM_BASE

                        # If using single layer, make sure we're not overwriting the input
                        if (not overwrite_ok) and in_map[offs >> 2]:
                            print(f'Layer {ll} output for CHW={c},{row},{col} is overwriting '
                                  f'input at location {offs:08x}')
                            if not no_error_stop:
                                sys.exit(1)
                        if out_map[offs >> 2]:
                            print(f'Layer {ll} output for CHW={c},{row},{col} is overwriting '
                                  f'itself at location {offs:08x}')
                            if not no_error_stop:
                                sys.exit(1)
                        out_map[offs >> 2] = True
                        apb_write(offs, val, rv=True)

            if not block_mode:
                memfile.write('  return rv;\n}\n')

        input_size = [out_size[0], out_size[1], out_size[2]]
        data = out_buf.reshape(input_size[0], input_size[1], input_size[2])
        in_map = out_map

    # Create run_test.sv
    with open(os.path.join(base_directory, test_name, runtest_filename), mode='w') as runfile:
        if block_mode:
            runfile.write('// Check default register values.\n')
            runfile.write('// Write all registers.\n')
            runfile.write('// Make sure only writable bits will change.\n')
            runfile.write('int     inp1;\n')
            runfile.write('string  fn;\n\n')
            if timeout:
                runfile.write(f'defparam REPEAT_TIMEOUT = {timeout};\n\n')
            runfile.write('initial begin\n')
            runfile.write('  //----------------------------------------------------------------\n')
            runfile.write('  // Initialize the CNN\n')
            runfile.write('  //----------------------------------------------------------------\n')
            runfile.write('  #200000;\n')
            runfile.write(f'  fn = {{`TARGET_DIR,"/{input_filename}.mem"}};\n')
            runfile.write('  inp1=$fopen(fn, "r");\n')
            runfile.write('  if (inp1 == 0) begin\n')
            runfile.write('    $display("ERROR : CAN NOT OPEN THE FILE");\n')
            runfile.write('  end\n')
            runfile.write('  else begin\n')
            runfile.write('    write_cnn(inp1);\n')
            runfile.write('    $fclose(inp1);\n')
            runfile.write('  end\n')
            runfile.write('end\n\n')
            runfile.write('initial begin\n')
            runfile.write('  #1;\n')
            runfile.write('  error_count = 0;\n')
            runfile.write('  @(posedge rstn);\n')
            runfile.write('  #5000;     // for invalidate done\n')
            runfile.write('  -> StartTest;\n')
            runfile.write('end\n')
        else:
            runfile.write(f'// {runtest_filename}\n')
            runfile.write('`define ARM_PROG_SOURCE test.c\n')
            if timeout:
                runfile.write(f'defparam REPEAT_TIMEOUT = {timeout};\n\n')

    return test_name


def main():
    """
    Command line wrapper
    """
    np.set_printoptions(threshold=np.inf, linewidth=190)

    parser = argparse.ArgumentParser(
        description="Tornado Memory Pooling and Convolution Simulation Test Data Generator")
    parser.add_argument('--apb-base', type=lambda x: int(x, 0), default=APB_BASE, metavar='N',
                        help=f"APB base address (default: {APB_BASE:08x})")
    parser.add_argument('--autogen', default='tests', metavar='S',
                        help="directory location for autogen_list (default: 'tests')")
    parser.add_argument('--c-filename', default='test', metavar='S',
                        help="C file name base (default: 'test' -> 'input.c')")
    parser.add_argument('--c-library', action='store_true',
                        help="use C library functions such as memset()")
    parser.add_argument('-D', '--debug', action='store_true',
                        help="debug mode (default: false)")
    parser.add_argument('--debug-computation', action='store_true',
                        help="debug computation (default: false)")
    parser.add_argument('--config-file', required=True, metavar='S',
                        help="YAML configuration file containing layer configuration")
    parser.add_argument('--checkpoint-file', required=True, metavar='S',
                        help="checkpoint file containing quantized weights")
    parser.add_argument('--input-filename', default='input', metavar='S',
                        help="input .mem file name base (default: 'input' -> 'input.mem')")
    parser.add_argument('--output-filename', default='output', metavar='S',
                        help="output .mem file name base (default: 'output' -> 'output-X.mem')")
    parser.add_argument('--runtest-filename', default='run_test.sv', metavar='S',
                        help="run test file name (default: 'run_test.sv')")
    parser.add_argument('--log-filename', default='log.txt', metavar='S',
                        help="log file name (default: 'log.txt')")
    parser.add_argument('--no-error-stop', action='store_true',
                        help="do not stop on errors (default: stop)")
    parser.add_argument('--input-offset', type=lambda x: int(x, 0), default=0,
                        metavar='N', choices=range(4*MEM_SIZE),
                        help="input offset (x8 hex, defaults to 0x0000)")
    parser.add_argument('--overwrite-ok', action='store_true',
                        help="allow output to overwrite input (default: warn/stop)")
    parser.add_argument('--queue-name', default='lowp', metavar='S',
                        help="queue name (default: 'lowp')")
    parser.add_argument('-L', '--log', action='store_true',
                        help="redirect stdout to log file (default: false)")
    parser.add_argument('--input-split', type=int, default=1, metavar='N',
                        choices=range(1, MAX_CHANNELS+1),
                        help="split input into N portions (default: don't split)")
    parser.add_argument('--stop-after', type=int, metavar='N',
                        help="stop after layer")
    parser.add_argument('--prefix', metavar='DIR', required=True,
                        help="set test name prefix")
    parser.add_argument('--test-dir', metavar='DIR',
                        help="set base directory name for auto-filing .mem files")
    parser.add_argument('--top-level', default=None, metavar='S',
                        help="top level name instead of block mode (default: None)")
    parser.add_argument('--timeout', type=int, metavar='N', default=10,
                        help="set timeout (units of 10ms, default 100ms)")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="verbose output (default: false)")
    parser.add_argument('--verify-writes', action='store_true',
                        help="verify write operations (toplevel only, default: false)")
    parser.add_argument('--zero-unused', action='store_true',
                        help="zero unused registers (default: do not touch)")
    args = parser.parse_args()

    if not args.test_dir:
        parser.error(f"Please specify output directory using --test-dir")

    # Load configuration file
    with open(args.config_file) as cfg_file:
        print(f'Reading {args.config_file} to configure network...')
        cfg = yaml.load(cfg_file, Loader=yaml.SafeLoader)

    if bool(set(cfg) - set(['dataset', 'layers', 'output_map', 'arch'])):
        print(f'Configuration file {args.config_file} contains unknown key(s)')
        sys.exit(1)

    cifar = 'dataset' in cfg and cfg['dataset'].lower() == 'cifar-10'
    if 'layers' not in cfg or 'arch' not in cfg:
        print(f'Configuration file {args.config_file} does not contain `layers` or `arch`')
        sys.exit(1)

    padding = []
    pool = []
    pool_stride = []
    output_offset = []
    processor_map = []
    average = []
    relu = []
    big_data = []

    for ll in cfg['layers']:
        if bool(set(ll) - set(['max_pool', 'avg_pool', 'pool_stride', 'out_offset',
                               'activate', 'data_format', 'processors', 'pad'])):
            print(f'Configuration file {args.config_file} contains unknown key(s) for `layers`')
            sys.exit(1)

        padding.append(ll['pad'] if 'pad' in ll else 1)
        if 'max_pool' in ll:
            pool.append(ll['max_pool'])
            average.append(0)
        elif 'avg_pool' in ll:
            pool.append(ll['avg_pool'])
            average.append(1)
        else:
            pool.append(0)
            average.append(0)
        pool_stride.append(ll['pool_stride'] if 'pool_stride' in ll else 0)
        output_offset.append(ll['out_offset'] if 'out_offset' in ll else 0)
        if 'processors' not in ll:
            print('`processors` key missing for layer in YAML configuration')
            sys.exit(1)
        processor_map.append(ll['processors'])
        if 'activate' in ll:
            if ll['activate'].lower() == 'relu':
                relu.append(1)
            else:
                print('unknown value for `activate` in YAML configuration')
                sys.exit(1)
        else:
            relu.append(0)
        if 'data_format' in ll:
            if big_data:  # Sequence not empty
                print('`data_format` can only be configured for the first layer')
                sys.exit(1)

            df = ll['data_format'].lower()
            if df in ['hwc', 'little']:
                big_data.append(False)
            elif df in ['chw', 'big']:
                big_data.append(True)
            else:
                print('unknown value for `data_format` in YAML configuration')
                sys.exit(1)
        else:
            big_data.append(False)

    if any(p < 0 or p > 2 for p in padding):
        print('Unsupported value for `pad` in YAML configuration')
        sys.exit(1)
    if any(p & 1 != 0 or p < 0 or p > 4 for p in pool):
        print('Unsupported value for `max_pool`/`avg_pool` in YAML configuration')
        sys.exit(1)
    if any(p == 3 or p < 0 or p > 4 for p in pool_stride):
        print('Unsupported value for `pool_stride` in YAML configuration')
        sys.exit(1)
    if any(p < 0 or p > 4*MEM_SIZE for p in output_offset):
        print('Unsupported value for `out_offset` in YAML configuration')
        sys.exit(1)

    # We don't support changing the following, but leave as parameters
    dilation = [1, 1]
    kernel_size = [3, 3]

    weights = []
    bias = []

    # Load weights and biases. This also configures the network channels.
    checkpoint = torch.load(args.checkpoint_file, map_location='cpu')
    print(f'Reading {args.checkpoint_file} to configure network weights...')

    if 'state_dict' not in checkpoint or 'arch' not in checkpoint:
        raise RuntimeError("\nNo `state_dict` or `arch` in checkpoint file.")

    if checkpoint['arch'].lower() != cfg['arch'].lower():
        print(f"Network architecture of configuration file ({cfg['arch']}) does not match "
              f"network architecture of checkpoint file ({checkpoint['arch']})")
        sys.exit(1)

    checkpoint_state = checkpoint['state_dict']
    layers = 0
    output_channels = []
    for _, k in enumerate(checkpoint_state.keys()):
        operation, parameter = k.rsplit(sep='.', maxsplit=1)
        if parameter in ['weight']:
            module, _ = k.split(sep='.', maxsplit=1)
            if module != 'fc':
                w = checkpoint_state[k].numpy().astype(np.int64)
                if layers == 0:
                    output_channels.append(w.shape[1])  # Input channels
                output_channels.append(w.shape[0])
                weights.append(w.reshape(-1, kernel_size[0], kernel_size[1]))
                layers += 1
                # Is there a bias for this layer?
                bias_name = operation + '.bias'
                if bias_name in checkpoint_state:
                    w = checkpoint_state[bias_name].numpy().astype(np.int64) // 128
                    bias.append(w)
                else:
                    bias.append(None)

    if layers != len(cfg['layers']):
        print('Number of layers in the YAML configuration file does not match the checkpoint file')
        sys.exit(1)

    if 'output_map' in cfg:
        # Use optional configuration value
        output_map = cfg['output_map']
    else:
        # Default to packed, 0-aligned output map
        output_map = 2**output_channels[layers]-1

    if popcount(output_map) != output_channels[layers]:
        print(f'The output_map ({output_map:016x}) does not correspond to the number of output '
              f'channels of the final layer ({output_channels[layers-1]}).')
        sys.exit(1)

    # We don't support changing the following, but leave as parameters
    stride = [1] * layers

    activate = [bool(x) for x in relu]
    pool_average = [bool(x) for x in average]

    data = sampledata.get(cifar)
    input_size = list(data.shape)

    if args.stop_after is not None:
        layers = args.stop_after + 1

    tn = create_sim(args.prefix, args.verbose,
                    args.debug, args.debug_computation, args.no_error_stop,
                    args.overwrite_ok, args.log, args.apb_base, layers, processor_map,
                    input_size, kernel_size, output_channels, padding, dilation, stride,
                    pool, pool_stride, pool_average, activate,
                    data, weights, bias, big_data, output_map,
                    args.input_split,
                    args.input_offset, output_offset,
                    args.input_filename, args.output_filename, args.c_filename,
                    args.test_dir, args.runtest_filename, args.log_filename,
                    args.zero_unused, args.timeout, not args.top_level, args.verify_writes,
                    args.c_library)

    # Append to regression list?
    if not args.top_level:
        testname = f'tests/{tn}/run_test:{args.queue_name}'
    else:
        testname = f'tests/{args.top_level}/{tn}/run_test:{args.queue_name}'
    found = False
    try:
        with open(os.path.join(args.autogen, 'autogen_list'), mode='r') as listfile:
            for line in listfile:
                if testname in line:
                    found = True
                    break
    except FileNotFoundError:
        pass
    if not found:
        with open(os.path.join(args.autogen, 'autogen_list'), mode='a') as listfile:
            listfile.write(f'{testname}\n')


def signal_handler(_signal, _frame):
    """
    Ctrl+C handler
    """
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()
