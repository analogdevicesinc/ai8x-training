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
import torch

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
P_NUMPRO = 16

INSTANCE_SIZE = 1024  # x32
TILE_SIZE = 0x40000
MEM_SIZE = INSTANCE_SIZE*16  # x32
MAX_DATA_DIM = 512  # Max theoretical size of single channel, used for command line validation only
MAX_CHANNELS = P_NUMPRO*P_NUMTILES


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
                                          f'wt_offs={wt_offs}: new val = {val}')

                val += bias[k]
                output[k][out_offs] = val
                if debug:
                    print(f'+bias {bias[k]} --> output[{k}][{out_offs}] = {val}')
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
               apb_base, layers, first_channel,
               input_size, kernel_size, chan, padding, dilation, stride,
               pool, pool_stride, pool_average, activate,
               data, kernel, bias, layer_has_bias, big_data, split,
               in_offset, out_offset, mem_instance, mem_tile,
               input_filename, output_filename, c_filename,
               base_directory, runtest_filename, log_filename,
               seed, zero_unused, timeout, block_mode, verify_writes):
    """
    Chain multiple CNN layers, create and save input and output
    """
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
        Write address and data to .mem file
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
                              f'rv = 0;{comment}\n')
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
        memfile.write(f'// Configuring input for {layers} layer(s), random seed={seed}')

        for ll in range(layers):
            memfile.write(f'// Layer {ll+1}: {chan[ll]}x{dim[ll][0]}x{dim[ll][1]} '
                          f'{"(big data)" if big_data[ll] else "(little data)"}, ')
            if pool[ll] > 0:
                memfile.write(f'{pool[ll]}x{pool[ll]} {"avg" if pool_average[ll] else "max"} '
                              f'pool with stride {pool_stride[ll]}')
            else:
                memfile.write(f'no pooling')
            memfile.write(f', {kernel_size[0]}x{kernel_size[1]} convolution '
                          f'with stride {stride[ll]} '
                          f'pad {padding[ll]}, {chan[ll+1]}x{dim[ll+1][0]}x{dim[ll+1][1]} out\n')

        if not block_mode:
            memfile.write('#include "global_functions.h"\n')
            memfile.write('#include <stdlib.h>\n')
            memfile.write('#include <stdint.h>\n\n')
            memfile.write('int cnn_check(void);\n\n')
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

        def apb_write_kern(tile, proc, idx, k):
            """
            Write kernel to .mem file
            """
            addr = C_TILE_OFFS*tile + C_MRAM_BASE + proc * 2**P_MASKABITS * 16 + idx * 16
            apb_write(addr, k[0] & 0xff, no_verify=True)
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

        # Initialize CNN registers

        # Calculate the number of tiles needed for each layer
        tiles = [min(P_NUMTILES,
                     max(chan[ll] if big_data[ll] else (chan[ll]+P_NUMPRO-1) // P_NUMPRO,
                         (chan[ll+1]+P_NUMPRO-1) // P_NUMPRO))
                 for ll in range(layers)]
        needed_tiles = max(tiles)  # Maximum tiles needed across whole network

        # Distribute input channels across tiles for each layer
        tiled_chan = [0] * layers
        for ll in range(layers):
            tiled_chan[ll] = [0] * needed_tiles
            if big_data[ll]:
                # Distribute 1 each into each tile until no more
                for tile in range(tiles[ll]):
                    tiled_chan[ll][tile] = (chan[ll] + P_NUMTILES - 1 - tile) // P_NUMTILES
            else:
                # 16 into each tile, remainder in last needed tile, then zeros
                for tile in range((chan[ll] - 1) // P_NUMPRO):
                    tiled_chan[ll][tile] = P_NUMPRO
                tiled_chan[ll][(chan[ll] - 1) // P_NUMPRO] = (chan[ll] - 1) % P_NUMPRO + 1

        if debug:
            print(f'channels   = {chan}')
            print(f'tiles      = {tiles}')
            print(f'max tiles  = {needed_tiles}')
            print(f'tiled_chan = {tiled_chan}')

        # Disable completely unused tiles
        for tile in range(needed_tiles, P_NUMTILES):
            apb_write_ctl(tile, 0, 0, comment=f' // disable tile {tile}')

        # Configure global control registers for used tiles
        for tile in range(needed_tiles):
            # Zero out Tornado RAM ('zero_tram')
            for p in range(P_NUMPRO):
                for offs in range(2**P_TRAMABITS):
                    apb_write_tram(tile, p, offs, 0, comment='zero ')

            # Stop state machine - will be overwritten later
            apb_write_ctl(tile, 0, 0x06, comment=' // stop SM')  # ctl reg
            # SRAM Control - does not need to be changed
            apb_write_ctl(tile, 1, 0x40e, comment=' // SRAM ctl')  # sram reg
            # Number of layers
            apb_write_ctl(tile, 2, layers-1, comment=' // layer count')  # layer max count reg

        for tile in range(needed_tiles):
            # Kernels ('load_mask')
            # Write only the kernels needed
            max_ptr = 0
            for ll in range(layers):
                for p in range(tiled_chan[ll][tile]):
                    # Stack kernels
                    for i in range(chan[ll+1]):
                        if big_data[ll]:
                            if tile < chan[ll]:  # Transpose kernels into tiles
                                k = kernel[ll][i*chan[ll] + tile].flatten()
                            else:  # Extra tile for output, use zero kernel
                                k = np.zeros(kernel_size[0] * kernel_size[1], dtype=np.int64)
                        else:
                            if tile < tiles[ll]:  # Transpose kernels
                                prev = sum(tiled_chan[ll][n] for n in range(tile))
                                k = kernel[ll][i*chan[ll] + prev + p].flatten()
                            else:
                                k = np.zeros(kernel_size[0] * kernel_size[1], dtype=np.int64)

                        if debug:
                            print(f'T{tile} L{ll} p{p+1}/{tiled_chan[ll][tile]} '
                                  f'm{i+1}/{chan[ll+1]}: {k}')
                        apb_write_kern(tile, p, max_ptr + i, k)
                max_ptr += chan[ll+1]

            # Bias ('zero_bias_ram')
            offs = 0
            i = 0
            ll = 0
            for _ in range(2**P_BRAMABITS):
                if ll < layers:
                    apb_write_bias(tile, offs, bias[ll][i])
                    i += 1
                    if i >= chan[ll+1]:
                        ll += 1
                        i = 0
                else:
                    apb_write_bias(tile, offs, 0)
                offs += 1

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
        for tile in range(needed_tiles):
            bias_ptr = 0
            mask_ptr = 0

            for ll in range(layers):
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

                # Configure SRAM write pointer ('config_cnn_wptr')
                apb_write_reg(tile, ll, 6, out_offset[ll+1] // 4, debug,
                              comment=' // SRAM write ptr')

                # Configure write pointer mask offset count ('config_cnn_woff')
                # [15:0]  Timeslot offset
                #         [11:0]  12 bits for memory - word address every time we reach mask limit
                #         [13:12] instance in group
                #         [15:14] by-16 group
                # [31:16] Mask offset (0x10000000, required when writing more than 4 masks)
                if chan[ll] * chan[ll+1] > 4:
                    val = 0x10000000
                else:
                    val = 0
                apb_write_reg(tile, ll, 7, val, debug, comment=' // mask offset count')

                # Configure sram read ptr count ('config_cnn_rptr')
                # Source address must match write pointer of previous layer
                apb_write_reg(tile, ll, 8, out_offset[ll] // 4, comment=' // SRAM read ptr')

                # Configure per-layer control
                # [3:0] s_slave: enable the by-4 group within the by-16 mask RAM to slave
                #                to first input volume; also enable timeslot
                # [4]   m_slave: slaves to 16x masters
                # [5]   master: sums all 16 processor outputs (vs 4 sums)
                # [6]   parallel: equals big data (per layer control)
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
                if tile == 0:
                    # Set external source for other active processing tiles
                    # (will be zero if only this tile is processing, never set bit 12)
                    val |= (2**min(P_NUMTILES-1, chan[ll]-1, tiles[ll]-1) - 1) << 13
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
                val = bias_ptr << 16 | mask_ptr << 8 | chan[ll+1]-1
                if tile == 0:
                    val |= 0x1000000  # Enable bias only for one tile
                apb_write_reg(tile, ll, 10, val, debug)
                bias_ptr += chan[ll+1]  # Each layer has output_channel number of bias values
                mask_ptr += chan[ll+1]

                # Configure tram pointer max ('config_cnn_tptr')
                if pool[ll] > 0:
                    val = max(0, (dim[ll][1] + pool_stride[ll] - pool[ll]) // pool_stride[ll] +
                              2*padding[ll] - 3)
                else:
                    val = max(0, dim[ll][1] + 2*padding[ll] - 3)
                apb_write_reg(tile, ll, 11, val, comment=' // TRAM ptr max')

                # Configure mask count and offset registers ('config_cnn_mena')
                # [15:0]  mask enable
                # [31:16] processor enable (or the reverse?)
                # When the input data is sources from 16 independent byte streams, all 16
                # processors and compute elements need to be enabled.  If there were only 4 input
                # channels, 0x000f000f would be correct.
                #
                # Enable at most 16 processors and masks
                if big_data[ll]:
                    bits = 2**((min(P_NUMPRO, chan[ll]) + tiles[ll]-1) // tiles[ll]) - 1
                else:
                    # Set only if needed for input channels, otherwise clear
                    bits = 2**min(P_NUMPRO, max(0, chan[ll] - tile*P_NUMPRO)) - 1
                apb_write_reg(tile, ll, 12, bits << 16 | bits, debug)

            if zero_unused:
                for ll in range(layers, C_MAX_LAYERS):
                    apb_write_reg(tile, ll, 0, 0, comment=' // zero unused')
                    apb_write_reg(tile, ll, 1, 0, comment=' // zero unused')
                    apb_write_reg(tile, ll, 3, 0, comment=' // zero unused')
                    apb_write_reg(tile, ll, 4, 0, comment=' // zero unused')
                    apb_write_reg(tile, ll, 5, 0, comment=' // zero unused')
                    apb_write_reg(tile, ll, 6, 0, comment=' // zero unused')
                    apb_write_reg(tile, ll, 7, 0, comment=' // zero unused')
                    apb_write_reg(tile, ll, 8, 0, comment=' // zero unused')
                    apb_write_reg(tile, ll, 9, 0, comment=' // zero unused')
                    apb_write_reg(tile, ll, 10, 0, comment=' // zero unused')
                    apb_write_reg(tile, ll, 11, 0, comment=' // zero unused')
                    apb_write_reg(tile, ll, 12, 0, comment=' // zero unused')

            # Load data memory ('admod_sram'/'lildat_sram')
            data_offs = C_TILE_OFFS*tile + C_SRAM_BASE
            if big_data[0]:
                if tile < chan[0]:
                    memfile.write(f'\n  // Big data input: {dim[0][0]}x{dim[0][1]}, '
                                  f'channel {tile+1} of {chan[0]}\n')
                    # "Big Data Mode" - Channels in sequence
                    chunk = input_size[1] // split
                    for c in range(tile, tile+(input_size[0] + tiles[0]-1) // tiles[0]):
                        # New channel - round up to next instance
                        data_offs = ((data_offs + mem_instance - 1) // mem_instance) * mem_instance
                        # (Note: We do not need to flush here, since that is done at the
                        # end of each channel's output below)
                        if debug:
                            print(f'T{tile} L0 data_offs:      {data_offs:08x}')
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
                                new_data_offs = ((data_offs + mem_instance - 1) //
                                                 mem_instance) * mem_instance
                                if new_data_offs != data_offs:
                                    apb_write_byte_flush(0)
                                data_offs = new_data_offs
                        if split > 1:
                            # Add bottom pad
                            for _ in range(padding[0]):
                                for _ in range(input_size[2]):
                                    apb_write_byte(data_offs, 0)
                                    data_offs += 1
                        apb_write_byte_flush(0)
            else:
                # "Little Data" - Four channels packed into a word
                for c in range(tile*P_NUMPRO, min(chan[0], (tile+1)*P_NUMPRO), 4):
                    memfile.write(f'\n  // Little data input: {dim[0][0]}x{dim[0][1]}, channels '
                                  f'{c+1} to {c+4} ({chan[0]} inputs)\n')
                    # Round up to next instance
                    data_offs = ((data_offs + 0x10*mem_instance-1) // (0x10*mem_instance)) * \
                        0x10*mem_instance
                    if debug:
                        print(f'T{tile} L0 data_offs:      {data_offs:08x}')
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
                    apb_write_byte_flush(0)

        for tile in range(1, needed_tiles):
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
            apb_write_ctl(tile, 0, 0x807, comment=' // enable')  # ctl reg

        # Master control - go
        apb_write_ctl(0, 0, 0x07, comment=' // master enable')  # ctl reg

        if not block_mode:
            memfile.write('  return 1;\n}\n\nint main(void)\n{\n  icache_enable();\n')
            memfile.write('  MXC_GCR->perckcn1 &= ~0x20; // Enable AI clock\n')
            memfile.write('  if (!cnn_load()) { fail(); return 0; }\n  cnn_wait();\n')
            memfile.write('  if (!cnn_check()) { fail(); return 0; }\n')
            memfile.write('  pass();\n  return 0;\n}\n\n')

        # End of input

    in_map = apb_write_byte_flush.mem

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

        # Write memfile for output
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
            for row in range(out_size[1]):
                for col in range(out_size[2]):
                    for c in range(0, chan[ll+1], 4):
                        val = out_buf[c][row][col] & 0xff
                        if c+1 < chan[ll+1]:
                            val |= (out_buf[c+1][row][col] & 0xff) << 8
                        if c+2 < chan[ll+1]:
                            val |= (out_buf[c+2][row][col] & 0xff) << 16
                        if c+3 < chan[ll+1]:
                            val |= (out_buf[c+3][row][col] & 0xff) << 24

                        offs = out_offset[ll+1] + ((c % 16)*mem_instance + (c // 16)*mem_tile +
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
    parser.add_argument('-a', '--average-pooling', type=int, action='append', metavar='N',
                        help="average pooling (all default to 0=max pooling)")
    parser.add_argument('--apb-base', type=lambda x: int(x, 0), default=0, metavar='N',
                        help="APB base address (default: 0)")
    parser.add_argument('--autogen', default='tests', metavar='S',
                        help="directory location for autogen_list (default: 'tests')")
    parser.add_argument('--c-filename', default='test', metavar='S',
                        help="C file name base (default: 'test' -> 'input.c')")
    parser.add_argument('-D', '--debug', action='store_true',
                        help="debug mode (default: false)")
    parser.add_argument('--debug-computation', action='store_true',
                        help="debug computation (default: false)")
    parser.add_argument('--input-filename', default='input', metavar='FN',
                        help="input .mem file name base (default: 'input' -> 'input.mem')")
    parser.add_argument('--output-filename', default='output', metavar='FN',
                        help="output .mem file name base (default: 'output' -> 'output-X.mem')")
    parser.add_argument('--runtest-filename', default='run_test.sv', metavar='FN',
                        help="run test file name (default: 'run_test.sv')")
    parser.add_argument('--log-filename', default='log.txt', metavar='FN',
                        help="log file name (default: 'log.txt')")
    parser.add_argument('--no-error-stop', action='store_true',
                        help="do not stop on errors (default: stop)")
    parser.add_argument('--input-offset', type=lambda x: int(x, 0), default=0,
                        metavar='N', choices=range(4*MEM_SIZE),
                        help="input offset (x8 hex, defaults to 0x0000)")
    parser.add_argument('-o', '--output-offset', type=lambda x: int(x, 0), action='append',
                        metavar='N',
                        help="output offset for each layer (x8 hex, all default to 0x0000)")
    parser.add_argument('--overwrite-ok', action='store_true',
                        help="allow output to overwrite input (default: warn/stop)")
    parser.add_argument('--queue-name', default='medium', metavar='S',
                        help="queue name (default: 'medium')")
    parser.add_argument('-L', '--log', action='store_true',
                        help="redirect stdout to log file (default: false)")
    parser.add_argument('-c', '--channel-start', type=int, action='append', metavar='N',
                        help="per-layer first channel (all default to 0)")
    parser.add_argument('-p', '--pad', type=int, action='append', metavar='N',
                        help="padding for each layer (all default to 1)")
    parser.add_argument('-x', '--pool', type=int, action='append', metavar='N',
                        help="pooling for each layer (all default to 2)")
    parser.add_argument('--pool-stride', type=int, action='append', metavar='N',
                        help="pooling stride for each layer (all default to 2)")
    parser.add_argument('--little-data', action='store_true',
                        help="little data input (default: big data = channels in sequence)")
    parser.add_argument('-r', '--relu', type=int, action='append', metavar='N',
                        help="activate layer using ReLU (all default to 0=no activation)")
    parser.add_argument('--input-split', type=int, default=1, metavar='N',
                        choices=range(1, MAX_CHANNELS+1),
                        help="split input into N portions (default: don't split)")
    parser.add_argument('--seed', type=int, metavar='N',
                        help="set fixed random seed")
    parser.add_argument('--stop-after', type=int, metavar='N',
                        help="stop after layer")
    parser.add_argument('--prefix', metavar='DIR', required=True,
                        help="set test name prefix")
    parser.add_argument('--test-dir', metavar='DIR',
                        help="set base directory name for auto-filing .mem files")
    parser.add_argument('--top-level', default=None, metavar='S',
                        help="top level name instead of block mode (default: None)")
    parser.add_argument('--timeout', type=int, metavar='N',
                        help="set timeout (units of 10ms, default 10ms)")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="verbose output (default: false)")
    parser.add_argument('--verify-writes', action='store_true',
                        help="verify write operations (toplevel only, default: false)")
    parser.add_argument('--zero-unused', action='store_true',
                        help="zero unused registers (default: do not touch)")
    args = parser.parse_args()
    if args.seed:
        np.random.seed(args.seed)

    if args.input_split != 1 and args.little_data:
        parser.error(f"--input-split is not supported for --little-data input")

    if not args.test_dir:
        parser.error(f"Please specify output directory using --test-dir")

    def check_arg(layers, arg, default, minimum, maximum, argstr):
        """
        Check a list of command line arguments
        """
        if arg:
            if any(a > maximum for a in arg) or any(a < minimum for a in arg):
                parser.error(f"all {argstr} values must be from {minimum} to {maximum}")
            if len(arg) != layers:
                parser.error(f"--layers is {layers}, must specify {layers} {argstr} values")
            val = arg
        else:
            val = [default] * layers
        return val

    # We don't support changing the following, but leave as parameters
    dilation = [1, 1]
    kernel_size = [3, 3]

    weights = []
    bias = []
    layer_has_bias = []

    # Load weights and biases. This also configures the network channels.
    checkpoint_file = 'ai84.pth.tar'
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    print(f'Reading {checkpoint_file} to configure network weights...')

    if 'state_dict' not in checkpoint:
        raise RuntimeError("\nNo state_dict in checkpoint file.")

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
                    bias.append(w.reshape(1))
                    layer_has_bias.append(True)
                else:
                    bias.append(np.repeat(np.asarray(0, dtype=np.int64), output_channels[-1]))
                    layer_has_bias.append(False)

    # We don't support changing the following, but leave as parameters
    stride = [1] * layers

    padding = check_arg(layers, args.pad, 1, 0, 2, '--pad')
    pool = check_arg(layers, args.pool, 2, 0, 4, '--pool')
    if any(p & 1 != 0 for p in pool):
        parser.error(f"unsupported value for --pool")
    pool_stride = check_arg(layers, args.pool_stride, 2, 0, 4, '--pool-stride')
    if any(p == 3 for p in pool_stride):
        parser.error(f"unsupported value for --pool-stride")
    output_offset = check_arg(layers, args.output_offset, 0, 0, 4*MEM_SIZE, '--output-offset')
    relu = check_arg(layers, args.relu, 0, 0, 1, '--relu')
    activate = [bool(x) for x in relu]
    average = check_arg(layers, args.average_pooling, 0, 0, 1, '--average-pooling')
    pool_average = [bool(x) for x in average]
    big_data = [False] * layers
    big_data[0] = not args.little_data
    # Distribute input channels across tiles for each layer
    first_channel = check_arg(layers, args.channel_start, 0, 0, MAX_CHANNELS-1,
                              '--channel-start')

    data = np.array([[[-128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
                       -127, -42, 5, -6, -2, 20, -18, -74, -128, -128,
                       -126, -128, -128, -127, -128, -128, -128, -128],
                      [-128, -128, -128, -128, -128, -128, -128, -123, -128, -70,
                       94, 110, 123, 127, 119, 116, 118, 116, 62, -128,
                       -128, -126, -127, -128, -128, -128, -128, -128],
                      [-128, -128, -128, -128, -128, -128, -127, -126, -128, 52,
                       107, 95, 96, 110, 112, 105, 109, 111, 84, -16,
                       -128, -126, -128, -128, -128, -128, -128, -128],
                      [-128, -128, -128, -128, -128, -128, -125, -128, -89, 57,
                       100, 101, 100, 97, 119, 113, 120, 84, 37, 61,
                       -128, -128, -127, -128, -128, -128, -128, -128],
                      [-128, -128, -128, -128, -128, -128, -124, -128, 31, 31,
                       39, 89, 115, 90, 30, 80, 48, 6, 20, 72,
                       -127, -128, -123, -127, -128, -128, -128, -128],
                      [-128, -128, -128, -128, -127, -128, -128, -128, 55, 61,
                       13, -4, 37, 65, 49, 1, -12, 16, 25, 62,
                       8, -120, -128, -126, -128, -128, -128, -128],
                      [-128, -128, -128, -128, -124, -128, -106, 47, 46, 40,
                       46, 47, 36, 56, 68, 23, 40, 41, 34, 30,
                       40, 79, -91, -128, -125, -128, -128, -128],
                      [-128, -128, -128, -128, -127, -128, 34, 55, 37, 39,
                       39, 34, 14, 65, 82, 11, 22, 20, 22, 29,
                       27, 45, 26, -128, -127, -128, -128, -128],
                      [-128, -128, -128, -128, -128, -113, 40, 36, 20, 20,
                       21, 20, 8, 43, 89, 9, 26, 24, 20, 14,
                       25, 28, 30, -104, -128, -128, -128, -128],
                      [-128, -128, -128, -128, -128, -65, 51, 28, 27, 40,
                       46, 45, 28, 39, 91, 8, 23, 24, 25, 39,
                       49, 22, 41, -55, -128, -128, -128, -128],
                      [-128, -128, -128, -128, -128, -22, 47, 45, 28, 27,
                       16, 16, 4, 24, 93, 3, 7, 9, 10, 13,
                       43, 33, 36, -18, -128, -128, -128, -128],
                      [-128, -128, -128, -128, -128, 27, 37, 42, 26, 31,
                       34, 38, 21, 24, 101, 15, 20, 32, 34, 38,
                       27, 14, 31, 28, -128, -128, -128, -128],
                      [-128, -128, -128, -128, -128, 47, 44, 57, 41, 29,
                       30, 21, 3, 7, 99, -2, -8, 4, 4, 5,
                       32, 49, 38, 40, -128, -128, -128, -128],
                      [-128, -128, -128, -128, -126, 46, 14, 30, 23, 14,
                       17, 22, 19, 18, 100, 17, 23, 25, 29, 28,
                       28, 37, 13, 41, -113, -128, -128, -128],
                      [-128, -128, -128, -128, -114, 62, 26, 65, 46, 37,
                       39, 42, 33, 15, 98, 18, 15, 24, 30, 27,
                       33, 46, 39, 52, -100, -128, -128, -128],
                      [-128, -128, -128, -128, -100, 51, 18, 34, 42, 15,
                       20, 18, 6, 1, 99, 11, -5, 2, 13, 8,
                       30, 41, -1, 34, -72, -128, -128, -128],
                      [-128, -128, -128, -128, -88, 65, 21, 34, 49, 27,
                       37, 32, 32, 16, 98, 22, 29, 33, 40, 27,
                       49, 46, 11, 52, -54, -128, -128, -128],
                      [-128, -128, -128, -128, -90, 54, 23, 33, 57, 30,
                       39, 29, 24, 3, 97, 17, -4, 3, 16, 28,
                       30, 62, 15, 42, -44, -128, -128, -128],
                      [-128, -128, -128, -128, -75, 64, 31, 27, 55, 38,
                       23, 18, 18, 2, 100, 22, 10, 18, 22, 32,
                       33, 72, 11, 50, -19, -128, -128, -128],
                      [-128, -128, -128, -128, -74, 57, 38, 43, 72, 24,
                       49, 53, 46, 21, 99, 30, 23, 36, 30, 21,
                       45, 72, 27, 44, -22, -128, -128, -128],
                      [-128, -128, -128, -128, -75, 54, 27, 28, 82, 29,
                       22, 25, 19, -1, 96, 30, -8, 5, 10, 26,
                       32, 59, 18, 46, -8, -128, -128, -128],
                      [-128, -128, -128, -128, -77, 63, 41, 42, 79, 42,
                       45, 45, 45, 19, 96, 39, 22, 32, 31, 31,
                       53, 68, 34, 52, -4, -128, -128, -128],
                      [-128, -128, -128, -128, -89, 50, 25, 25, 77, 38,
                       45, 44, 48, 24, 92, 44, 25, 32, 33, 30,
                       51, 61, 16, 32, -1, -128, -128, -128],
                      [-128, -128, -128, -128, -101, 65, 41, 29, 78, 32,
                       17, 22, 20, 3, 96, 45, -9, 8, 11, 28,
                       43, 90, 34, 43, 18, -128, -128, -128],
                      [-128, -128, -128, -128, -126, 56, 30, 25, 63, 42,
                       45, 48, 52, 27, 93, 60, 30, 44, 30, 34,
                       44, 54, 34, 45, 5, -128, -128, -128],
                      [-128, -128, -128, -128, -128, -4, 64, 59, 68, 25,
                       27, 32, 24, 10, 90, 54, 10, 33, 10, 23,
                       47, 67, 48, 56, -19, -128, -128, -128],
                      [-128, -128, -128, -128, -128, -128, -78, -97, -15, 78,
                       64, 75, 69, 46, 119, 87, 44, 64, 51, 70,
                       75, 8, 25, 17, -128, -128, -128, -128],
                      [-128, -128, -128, -128, -126, -128, -128, -128, -128, -74,
                       -17, -16, -8, -23, -11, -15, 8, -9, -29, -37,
                       -87, -128, -128, -128, -128, -128, -128, -128]]],
                    dtype=np.int64)
    input_size = list(data.shape)

    timeout = args.timeout
    # Double timeout for top level
    if args.top_level:
        if timeout:
            timeout *= 3
        else:
            timeout = 3

    if args.stop_after is not None:
        layers = args.stop_after + 1

    tn = create_sim(args.prefix, args.verbose,
                    args.debug, args.debug_computation, args.no_error_stop,
                    args.overwrite_ok, args.log, args.apb_base, layers, first_channel,
                    input_size, kernel_size, output_channels, padding, dilation, stride,
                    pool, pool_stride, pool_average, activate,
                    data, weights, bias, layer_has_bias, big_data,
                    args.input_split,
                    args.input_offset, output_offset,
                    INSTANCE_SIZE, TILE_SIZE,
                    args.input_filename, args.output_filename, args.c_filename,
                    args.test_dir, args.runtest_filename, args.log_filename,
                    args.seed,
                    args.zero_unused, timeout, not args.top_level, args.verify_writes)

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
