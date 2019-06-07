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
import os
import signal
import sys

import numpy as np
import tabulate
import torch
import yaml

import apbaccess
import commandline
import rtlsim
import sampledata
import toplevel
from tornadocnn import MAX_LAYERS, TRAM_SIZE, BIAS_SIZE, MASK_WIDTH, \
    C_TRAM_BASE, C_SRAM_BASE, C_GROUP_OFFS, \
    P_NUMGROUPS, P_NUMPRO, P_SHARED, INSTANCE_SIZE, GROUP_SIZE, MEM_SIZE, MAX_CHANNELS, \
    REG_CTL, REG_SRAM, REG_LCNT_MAX, \
    LREG_RCNT, LREG_CCNT, LREG_RFU, LREG_PRCNT, LREG_PCCNT, LREG_STRIDE, LREG_WPTR_BASE, \
    LREG_WPTR_OFFS, LREG_RPTR_BASE, LREG_LCTL, LREG_MCNT, LREG_TPTR, LREG_ENA, MAX_LREG, \
    BIAS_DIV, set_device
from simulate import cnn_layer
from utils import argmin, ffs, fls, popcount, s2u


def create_sim(prefix, verbose, debug, debug_computation, no_error_stop, overwrite_ok, log,
               apb_base, layers, processor_map,
               input_size, kernel_size, chan, padding, dilation, stride,
               pool, pool_stride, pool_average, activate,
               data, kernel, bias, big_data, output_map, split,
               in_offset, out_offset,
               input_filename, output_filename, c_filename,
               base_directory, runtest_filename, log_filename,
               zero_unused, timeout, block_mode, verify_writes,
               c_library=False,
               ai85=False):
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

    # Complete list of maps with output map
    processor_map.append(output_map)

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

    # Redirect stdout?
    if log:
        sys.stdout = open(os.path.join(base_directory, test_name, log_filename), 'w')
        print(f'{test_name}')

    if block_mode:
        filename = input_filename + '.mem'
    else:
        filename = c_filename + '.c'
    with open(os.path.join(base_directory, test_name, filename), mode='w') as memfile:
        apb = apbaccess.apbwriter(memfile, apb_base, block_mode, verify_writes, no_error_stop)

        memfile.write(f'// {test_name}\n')
        memfile.write(f'// Created using {" ".join(str(x) for x in sys.argv)}\n')

        # Human readable description of test
        memfile.write(f'\n// Configuring input for {layers} layer{"s" if layers > 1 else ""}\n')

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
            toplevel.header(memfile, apb_base, c_library)
            toplevel.load_header(memfile)

        # Calculate the groups needed, and groups and processors used overall
        processors_used = 0
        group_map = []
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
            for group in range(P_NUMGROUPS):
                if (processor_map[ll] >> group*P_NUMPRO) % 2**P_NUMPRO:
                    this_map.append(group)
            group_map.append(this_map)

        groups_used = []
        for group in range(P_NUMGROUPS):
            if ((processors_used | processor_map[layers]) >> group*P_NUMPRO) % 2**P_NUMPRO:
                groups_used.append(group)

        # Initialize CNN registers

        if verbose:
            print('\nGlobal registers:')
            print('-----------------')

        # Disable completely unused groups
        for group in range(P_NUMGROUPS):
            if group not in groups_used:
                apb.write_ctl(group, REG_CTL, 0,
                              verbose, comment=f' // Disable group {group}')

        memfile.write('\n')

        # Configure global control registers for used groups
        for _, group in enumerate(groups_used):
            # Zero out Tornado RAM
            if c_library:
                addr = apb_base + C_GROUP_OFFS*group + C_TRAM_BASE
                memfile.write(f'  memset((uint32_t *) 0x{addr:08x}, 0, '
                              f'{TRAM_SIZE * P_NUMPRO * 4}); // Zero TRAM group {group}\n')
            else:
                for p in range(P_NUMPRO):
                    for offs in range(TRAM_SIZE):
                        apb.write_tram(group, p, offs, 0, comment='Zero ')

            memfile.write('\n')

            # Stop state machine - will be overwritten later
            apb.write_ctl(group, REG_CTL, 0x06,
                          verbose, comment=' // Stop SM')
            # SRAM Control - does not need to be changed
            apb.write_ctl(group, REG_SRAM, 0x40e,
                          verbose, comment=' // SRAM control')
            # Number of layers
            apb.write_ctl(group, REG_LCNT_MAX, layers-1,
                          verbose, comment=' // Layer count')
            memfile.write('\n')

        def print_kernel_map(kmap):
            """
            Print map of all used kernels in kernel map `kmap`.
            """
            table = tabulate.tabulate(kmap, tablefmt='plain', missingval='X')
            print('-' * MASK_WIDTH)
            if layers < 10:
                print(table.replace('  ', ''))
            print('-' * MASK_WIDTH)

        # Kernels: Stack kernels; write only the kernels needed
        chan_kern_max = [0] * MAX_CHANNELS
        kern_offs = [0] * layers
        kern_len = [0] * layers
        kernel_map = [[None] * MASK_WIDTH for i in range(MAX_CHANNELS)]
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
            # need to be written, i.e. round down the first. The last does not need to be rounded
            # up because hardware takes care of it.
            next_layer_map = processor_map[ll+1]
            kern_len[ll] = 1 + fls(next_layer_map) - (ffs(next_layer_map) & ~(P_SHARED-1))

            # We don't have to use dummy columns if there's space available on the left
            kern_offs[ll] = max(0, kern_offs[ll] - (ffs(next_layer_map) % P_SHARED))

            # The kernel offset needs to start at a multiple of 4.
            kern_offs[ll] = (kern_offs[ll] + P_SHARED-1) & ~(P_SHARED-1)

            if kern_offs[ll] + kern_len[ll] > MASK_WIDTH:
                print(f'\nKernel memory exceeded at layer {ll}; offset: {kern_offs[ll]}, '
                      f'needed: {kern_len[ll]}.')
                print('\nKernel map so far:')
                print_kernel_map(kernel_map)
                sys.exit(1)

            for c in range(first_channel, last_channel+1):
                if (processor_map[ll] >> c) & 1 == 0:
                    # Unused processor
                    continue

                # Start at the first used instance
                this_map = next_layer_map >> ffs(next_layer_map)
                coffs = ffs(next_layer_map) % P_SHARED
                for col in range(chan[ll+1]):
                    # Skip over unused bits in the processor map
                    while this_map & 1 == 0:
                        assert this_map != 0
                        coffs += 1
                        this_map >>= 1
                    this_map >>= 1

                    k = kernel[ll][ch + col*chan[ll]].flatten()
                    if debug:
                        print(f'Channel {c} Layer {ll} m{col}/{chan[ll+1]-1}: {k}')
                    apb.write_kern(ll, c, kern_offs[ll] + col + coffs, k)

                    # Update kernel map
                    assert kernel_map[c][kern_offs[ll] + col + coffs] is None
                    kernel_map[c][kern_offs[ll] + col + coffs] = ll

                assert kern_len[ll] == coffs + chan[ll+1]
                chan_kern_max[c] = kern_offs[ll] + kern_len[ll]
                ch += 1

        if verbose:
            print('\nKernel map:')
            print_kernel_map(kernel_map)

        # Bias: Each group has one bias memory (size BIAS_SIZE bytes). Use only the bias memory in
        # one selected group for the layer, and only if the layer uses a bias. Keep track of the
        # offsets so they can be programmed into the mask count register later.
        group_bias_max = [0] * P_NUMGROUPS
        bias_offs = [None] * layers
        bias_group = [None] * layers
        for ll in range(layers):
            if bias[ll] is None:
                continue
            if len(bias[ll]) != chan[ll+1]:
                print(f'Layer {ll}: output channel count {chan[ll+1]} does not match the number '
                      f'of bias values {len(bias[ll])}.')
                sys.exit(1)

            # Pick the group with the least amount of data in it
            group = argmin(group_bias_max[t] for t in group_map[ll])
            if group_bias_max[group] + chan[ll+1] > BIAS_SIZE:
                print(f'Layer {ll}: bias memory capacity exceeded - available groups: '
                      f'{group_map[ll]}, used so far: {group_bias_max}, needed: {chan[ll+1]}.')
                sys.exit(1)

            bias_group[ll] = group
            bias_offs[ll] = group_bias_max[group]
            # Each layer has output_channel number of bias values
            for i in range(chan[ll+1]):
                apb.write_bias(group, bias_offs[ll] + i, bias[ll][i])
            group_bias_max[group] += chan[ll+1]

        if verbose:
            print('\nGlobal configuration:')
            print('---------------------')
            print(f'Used processors    = {processors_used:016x}')
            print(f'Used groups         = {groups_used}')
            print(f'Input offset       = {in_offset}')
            print('\nPer-group configuration:')
            print('-----------------------')
            print(f'Used bias memory   = {group_bias_max}')
            print('\nPer-layer configuration:')
            print('------------------------')
            print(f'Number of channels = {chan[:layers]} -> {chan[layers]} outputs')
            print('Processor map      = [',
                  ', '.join('{:016x}'.format(k) for k in processor_map[:layers]), ']',
                  f' -> {processor_map[layers]:016x} output', sep='',)
            print(f'Group map          = {group_map}')
            print(f'Kernel offsets     = {kern_offs}')
            print(f'Kernel lengths     = {kern_len}')
            print(f'Group with bias    = {bias_group}')
            print(f'Bias offsets       = {bias_offs}')
            print(f'Output offsets     = {out_offset}')
            print('')

        if verbose:
            print('Layer register configuration:')
            print('-----------------------------')

        # Configure per-layer control registers
        for _, group in enumerate(groups_used):
            for ll in range(layers):
                memfile.write(f'\n  // Group {group} layer {ll}\n')

                # Configure row count
                # [7:0] maxcount: lower 8 bits = total of width + pad - 1
                # [9:8] pad: 2 bits pad
                apb.write_lreg(group, ll, LREG_RCNT,
                               (padding[ll] << 8) | (dim[ll][0]-1 + 2*padding[ll]),
                               verbose, comment=' // Rows')

                # Configure column count
                # [7:0] width including padding - 1
                # [9:8] pad count (0 = no pad, 1 = half pad, 2 = full pad)
                apb.write_lreg(group, ll, LREG_CCNT,
                               padding[ll] << 8 | (dim[ll][1]-1 + 2 * padding[ll]),
                               verbose, comment=' // Columns')

                # Configure pooling row count
                apb.write_lreg(group, ll, LREG_PRCNT, max(1, pool[ll]-1),
                               verbose, comment=' // Pooling rows')

                # Configure pooling column count
                apb.write_lreg(group, ll, LREG_PCCNT, max(1, pool[ll]-1),
                               verbose, comment=' // Pooling columns')

                # Configure pooling stride count
                apb.write_lreg(group, ll, LREG_STRIDE, pool_stride[ll]-1,
                               verbose, comment=' // Pooling stride')

                # Configure SRAM write pointer -- write ptr is global
                # Get offset to first available instance of the first used processor of the next
                # layer.
                instance = ffs(processor_map[ll+1]) & ~(P_SHARED-1)
                apb.write_lreg(group, ll, LREG_WPTR_BASE, out_offset[ll] // 4 +
                               ((instance % P_SHARED) * INSTANCE_SIZE |
                                ((instance // P_SHARED) << 12)),
                               verbose, comment=' // SRAM write ptr')

                # Configure write pointer mask offset count
                # [15:0]  Timeslot offset
                #         [11:0]  12 bits for memory - word address every time we reach mask limit
                #         [13:12] instance in group
                #         [15:14] by-16 group
                # [31:16] Mask offset (0x10000000, required when writing more than 4 masks)
                if chan[ll] * kern_len[ll] > 4:
                    val = 0x10000000
                else:
                    val = 0
                apb.write_lreg(group, ll, LREG_WPTR_OFFS, val,
                               verbose, comment=' // Mask offset count')

                # Configure sram read ptr count -- read ptr is local
                # Source address must match write pointer of previous layer (minus global offset)
                apb.write_lreg(group, ll, LREG_RPTR_BASE,
                               in_offset // 4 if ll == 0 else out_offset[ll-1] // 4,
                               verbose, comment=' // SRAM read ptr')

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
                if group == group_map[ll][0]:
                    # Set external source for other active processing groups (can be zero if no
                    # other groups are processing). Do not set the bit corresponding to this group
                    # (e.g., if group == 0, do not set bit 12)
                    sources = 0
                    for t in range(group_map[ll][0]+1, P_NUMGROUPS):
                        # See if any processors other than this one are operating
                        # and set the cnnsiena bit if true
                        if (processor_map[ll] >> (t * P_NUMPRO)) % 2**P_NUMPRO:
                            sources |= 1 << t
                    val |= sources << 12
                apb.write_lreg(group, ll, LREG_LCTL, val,
                               verbose, comment=' // Layer control')

                # Configure mask count
                # Restriction: Every one of the mask memories will have to start from same offset
                # [6:0]   Max count (output channels)
                # [7]     RFU
                # [14:8]  Starting address for group of 16
                # [15]    RFU
                # [23:16] Bias pointer starting address
                # [24]    Bias enable
                # [31:25] RFU
                val = kern_offs[ll] << 8 | kern_len[ll]-1
                if group == bias_group[ll]:
                    # Enable bias only for one group
                    val |= 0x1000000 | bias_offs[ll] << 16
                apb.write_lreg(group, ll, LREG_MCNT, val,
                               verbose, comment=' // Mask offset and count')

                # Configure tram pointer max
                if pool[ll] > 0:
                    val = max(0, (dim[ll][1] + pool_stride[ll] - pool[ll]) // pool_stride[ll] +
                              2*padding[ll] - 3)
                else:
                    val = max(0, dim[ll][1] + 2*padding[ll] - 3)
                apb.write_lreg(group, ll, LREG_TPTR, val,
                               verbose, comment=' // TRAM ptr max')

                # Configure mask and processor enables
                # [15:0]  processor enable
                # [31:16] mask enable
                # When the input data is sourced from 16 independent byte streams, all 16
                # processors and compute elements need to be enabled.  If there were only 4 input
                # channels, 0x000f000f would be correct.
                #
                # Enable at most 16 processors and masks
                bits = (processor_map[ll] >> group*P_NUMPRO) % 2**P_NUMPRO
                apb.write_lreg(group, ll, LREG_ENA, bits << 16 | bits,
                               verbose, comment=' // Mask and processor enables')

            if zero_unused:
                for ll in range(layers, MAX_LAYERS):
                    for reg in range(MAX_LREG+1):
                        if reg == LREG_RFU:  # Register 2 not implemented
                            continue
                        apb.write_lreg(group, ll, reg, 0,
                                       verbose, comment=f' // Zero unused layer {ll} registers')

        # Load data memory
        # Start loading at the first used group
        memfile.write(f'\n\n  // {chan[0]}-channel data input\n')
        c = 0
        data_offs = 0
        step = 1 if big_data[0] else 4
        for ch in range(0, MAX_CHANNELS, step):
            if not (processor_map[0] >> ch) % 2**step:
                # Channel or block of four channels not used for input
                continue

            # Load channel into shared memory
            group = ch // P_NUMPRO
            instance = (ch % P_NUMPRO) // P_SHARED
            new_data_offs = C_GROUP_OFFS*group + C_SRAM_BASE + INSTANCE_SIZE*4*instance
            if new_data_offs == data_offs:
                print('Layer 0 processor map is misconfigured for data input. '
                      f'There is data overlap between processors {ch-1} and {ch}')
                sys.exit(1)
            data_offs = new_data_offs

            if debug:
                print(f'G{group} L0 data_offs:      {data_offs:08x}')

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
                            apb.write_byte(data_offs, 0)
                            data_offs += 1
                row = 0
                for s in range(split):
                    if split > 1 and s + 1 < split:
                        overlap = padding[0]
                    else:
                        overlap = 0
                    while row < (s + 1) * chunk + overlap:
                        for col in range(input_size[2]):
                            apb.write_byte(data_offs, s2u(data[c][row][col]))
                            data_offs += 1
                        row += 1
                    row -= 2*overlap  # Rewind
                    # Switch to next memory instance
                    if split > 1 and s + 1 < split:
                        new_data_offs = ((data_offs + INSTANCE_SIZE - 1) //
                                         INSTANCE_SIZE) * INSTANCE_SIZE
                        if new_data_offs != data_offs:
                            apb.write_byte_flush(0)
                        data_offs = new_data_offs
                if split > 1:
                    # Add bottom pad
                    for _ in range(padding[0]):
                        for _ in range(input_size[2]):
                            apb.write_byte(data_offs, 0)
                            data_offs += 1
                c += 1
            else:
                # HWC ("Little Data") - Four channels packed into a word (0BGR0BGR0BGR0BGR0BGR....)
                memfile.write(f'  // HWC (little data): {dim[0][0]}x{dim[0][1]}, '
                              f'channels {c} to {min(c+3, chan[0]-1)}\n')

                for row in range(input_size[1]):
                    for col in range(input_size[2]):
                        if c < chan[0]:
                            apb.write_byte(data_offs, s2u(data[c][row][col]))
                        else:
                            apb.write_byte(data_offs, 0)
                        data_offs += 1
                        # Always write multiple of four bytes even for last input
                        if c+1 < chan[0]:
                            apb.write_byte(data_offs, s2u(data[c+1][row][col]))
                        else:
                            apb.write_byte(data_offs, 0)
                        data_offs += 1
                        if c+2 < chan[0]:
                            apb.write_byte(data_offs, s2u(data[c+2][row][col]))
                        else:
                            apb.write_byte(data_offs, 0)
                        data_offs += 1
                        if c+3 < chan[0]:
                            apb.write_byte(data_offs, s2u(data[c+3][row][col]))
                        else:
                            apb.write_byte(data_offs, 0)
                        data_offs += 1
                c += 4

            apb.write_byte_flush(0)
            if c >= chan[0]:
                # Consumed all available channels
                break

        memfile.write(f'  // End of data input\n\n')

        if verbose:
            print('\nGlobal registers:')
            print('-----------------')

        # Enable all needed groups except the first one
        for _, group in enumerate(groups_used[1:]):
            # [0]    enable
            # [2:1]  rdy_sel  (wait states - set to max)
            # [3]    RFU
            # [4]    calcmax
            # [5]    poolena
            # [6]    bigdata
            # [7]    actena
            # [8]    one-shot (stop after single layer)
            # [11:9] ext_sync (slave to other group)
            # [12]   irq
            apb.write_ctl(group, REG_CTL, 0x807 | groups_used[0] << 9,
                          verbose, comment=f' // Enable group {group}')

        # Master control - go
        apb.write_ctl(groups_used[0], REG_CTL, 0x07,
                      verbose, comment=f' // Master enable group {groups_used[0]}')

        if not block_mode:
            toplevel.load_footer(memfile)
            toplevel.main(memfile)

        # End of input

    in_map = apb.get_mem()

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
                                      data,
                                      ai85=ai85,
                                      debug=debug_computation)

        # Write .mem file for output or create the C cnn_check() function to verify the output
        out_map = [False] * C_GROUP_OFFS * P_NUMGROUPS
        if block_mode:
            if ll == layers-1:
                filename = output_filename + '.mem'  # Final output
            else:
                filename = f'{output_filename}-{ll+1}.mem'  # Intermediate output
            filemode = 'w'
        else:
            if ll == layers-1:
                filename = c_filename + '.c'  # Final output
            else:  # FIXME: Make this None and don't rely on /dev/null
                filename = '/dev/null'  # Intermediate output - used for layer overwrite check
            filemode = 'a'
        with open(os.path.join(base_directory, test_name, filename), mode=filemode) as memfile:
            apb.set_memfile(memfile)

            if memfile is not None:
                memfile.write(f'// {test_name}\n// Expected output of layer {ll+1}\n')
                if not block_mode:
                    toplevel.verify_header(memfile)

            # Start at the instance of the first active output processor/channel
            coffs_start = ffs(processor_map[ll+1]) & ~(P_SHARED-1)
            next_layer_map = processor_map[ll+1] >> coffs_start

            for row in range(out_size[1]):
                for col in range(out_size[2]):
                    this_map = next_layer_map
                    coffs = coffs_start
                    c = 0
                    while c < chan[ll+1]:
                        # Get four bytes either from output or zeros and construct HWC word
                        val = 0
                        for _ in range(4):
                            val >>= 8
                            if this_map & 1:
                                val |= out_buf[c][row][col] << 24
                                c += 1
                            this_map >>= 1

                        # Physical offset into instance and group
                        proc = (coffs % MAX_CHANNELS) & ~(P_SHARED-1)
                        offs = C_SRAM_BASE + out_offset[ll] + \
                            (((proc % P_NUMPRO) * INSTANCE_SIZE |
                              (proc // P_NUMPRO) * GROUP_SIZE) +
                             row*out_size[2] + col) * 4

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
                        apb.verify(offs, val, rv=True)
                        coffs += 4

            if memfile is not None and not block_mode:
                toplevel.verify_footer(memfile)

        input_size = [out_size[0], out_size[1], out_size[2]]
        data = out_buf.reshape(input_size[0], input_size[1], input_size[2])
        in_map = out_map

    # Create run_test.sv
    rtlsim.create_runtest_sv(block_mode, base_directory, test_name, runtest_filename,
                             input_filename, timeout)

    return test_name


def main():
    """
    Command line wrapper
    """
    np.set_printoptions(threshold=np.inf, linewidth=190)

    args = commandline.get_parser()

    # Configure device
    set_device(args.ai85)

    # Load configuration file
    with open(args.config_file) as cfg_file:
        print(f'Reading {args.config_file} to configure network...')
        cfg = yaml.load(cfg_file, Loader=yaml.SafeLoader)

    if bool(set(cfg) - set(['dataset', 'layers', 'output_map', 'arch'])):
        print(f'Configuration file {args.config_file} contains unknown key(s).')
        sys.exit(1)

    cifar = 'dataset' in cfg and cfg['dataset'].lower() == 'cifar-10'
    if 'layers' not in cfg or 'arch' not in cfg:
        print(f'Configuration file {args.config_file} does not contain `layers` or `arch`.')
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
            print(f'Configuration file {args.config_file} contains unknown key(s) for `layers`.')
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
            print('`processors` key missing for layer in YAML configuration.')
            sys.exit(1)
        processor_map.append(ll['processors'])
        if 'activate' in ll:
            if ll['activate'].lower() == 'relu':
                relu.append(1)
            else:
                print('Unknown value for `activate` in YAML configuration.')
                sys.exit(1)
        else:
            relu.append(0)
        if 'data_format' in ll:
            if big_data:  # Sequence not empty
                print('`data_format` can only be configured for the first layer.')
                sys.exit(1)

            df = ll['data_format'].lower()
            if df in ['hwc', 'little']:
                big_data.append(False)
            elif df in ['chw', 'big']:
                big_data.append(True)
            else:
                print('Unknown value for `data_format` in YAML configuration.')
                sys.exit(1)
        else:
            big_data.append(False)

    if any(p < 0 or p > 2 for p in padding):
        print('Unsupported value for `pad` in YAML configuration.')
        sys.exit(1)
    if any(p & 1 != 0 or p < 0 or p > 4 for p in pool):
        print('Unsupported value for `max_pool`/`avg_pool` in YAML configuration.')
        sys.exit(1)
    if any(p == 3 or p < 0 or p > 4 for p in pool_stride):
        print('Unsupported value for `pool_stride` in YAML configuration.')
        sys.exit(1)
    if any(p < 0 or p > 4*MEM_SIZE for p in output_offset):
        print('Unsupported value for `out_offset` in YAML configuration.')
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
              f"network architecture of checkpoint file ({checkpoint['arch']}).")
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
                assert w.min() >= -128 and w.max() <= 127
                if layers == 0:
                    output_channels.append(w.shape[1])  # Input channels
                output_channels.append(w.shape[0])
                weights.append(w.reshape(-1, kernel_size[0], kernel_size[1]))
                layers += 1
                # Is there a bias for this layer?
                bias_name = operation + '.bias'
                if bias_name in checkpoint_state:
                    w = checkpoint_state[bias_name].numpy().astype(np.int64) // BIAS_DIV
                    assert w.min() >= -128 and w.max() <= 127
                    bias.append(w)
                else:
                    bias.append(None)

    if layers != len(cfg['layers']):
        print(f"Number of layers in the YAML configuration file ({len(cfg['layers'])}) "
              f"does not match the checkpoint file ({layers}).")
        sys.exit(1)

    if args.stop_after is not None:
        layers = args.stop_after + 1

    if 'output_map' in cfg:
        # Use optional configuration value
        output_map = cfg['output_map']
    else:
        if len(processor_map) > layers:
            output_map = processor_map[layers]
        else:
            # Default to packed, 0-aligned output map
            output_map = 2**output_channels[layers]-1

    if popcount(output_map) != output_channels[layers]:
        print(f'The output_map ({output_map:016x}) does not correspond to the number of output '
              f'channels of the final layer ({output_channels[layers]}).')
        sys.exit(1)

    # Remove extraneous input layer configurations (when --stop-after is used)
    processor_map = processor_map[:layers]
    output_channels = output_channels[:layers+1]
    output_offset = output_offset[:layers]

    # We don't support changing the following, but leave as parameters
    stride = [1] * layers

    activate = [bool(x) for x in relu]
    pool_average = [bool(x) for x in average]

    data = sampledata.get(cifar)
    input_size = list(data.shape)

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
                    args.c_library,
                    args.ai85)

    rtlsim.append_regression(args.top_level, tn, args.queue_name, args.autogen)


def signal_handler(_signal, _frame):
    """
    Ctrl+C handler
    """
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()
