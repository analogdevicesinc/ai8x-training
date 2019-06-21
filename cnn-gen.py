#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2018-2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
###################################################################################################
"""
Embedded network and simulation test generator program for Tornado CNN
"""
import os
import signal
import sys

import numpy as np

import apbaccess
import checkpoint
import cmsisnn
import commandline
import kernels
import load
import rtlsim
import sampledata
import stats
import tornadocnn as tc
import yamlcfg
from simulate import cnn_layer, linear_layer
from utils import ffs, popcount


def create_net(prefix, verbose, debug, debug_computation, no_error_stop, overwrite_ok, log,
               apb_base, layers, processor_map,
               input_size, kernel_size, quantization,
               chan, padding, dilation, stride,
               pool, pool_stride, pool_average, activate,
               data, kernel, bias, big_data, output_map, fc_weights, fc_bias,
               split,
               in_offset, out_offset,
               input_filename, output_filename, c_filename,
               base_directory, runtest_filename, log_filename,
               zero_unused, timeout, block_mode, verify_writes,
               embedded_code=False, weight_filename=None, sample_filename=None,
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
        dim.append([(pooled_size[0] - dilation[ll][0] * (kernel_size[ll][0] - 1) - 1 +
                     2 * padding[ll]) // stride[ll] + 1,
                    (pooled_size[1] - dilation[ll][1] * (kernel_size[ll][1] - 1) - 1 +
                     2 * padding[ll]) // stride[ll] + 1])

    # Complete list of maps with output map
    processor_map.append(output_map)

    # Create comment of the form "k1_b0-1x32x32b_2x2s2p14-..."
    test_name = prefix
    if not embedded_code:
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
    if embedded_code:
        sampledata_header = \
            open(os.path.join(base_directory, test_name, sample_filename), mode='w')
        weight_header = \
            open(os.path.join(base_directory, test_name, weight_filename), mode='w')
    else:
        sampledata_header = weight_header = None

    with open(os.path.join(base_directory, test_name, filename), mode='w') as memfile:
        apb = apbaccess.apbwriter(memfile, apb_base, block_mode, verify_writes, no_error_stop,
                                  weight_header=weight_header, sampledata_header=sampledata_header,
                                  embedded_code=embedded_code, ai85=ai85)

        apb.copyright_header()

        apb.output(f'// {test_name}\n')
        apb.output(f'// Created using {" ".join(str(x) for x in sys.argv)}\n')

        # Human readable description of test
        apb.output(f'\n// Configuring {layers} layer{"s" if layers > 1 else ""}:\n')

        for ll in range(layers):
            apb.output(f'// Layer {ll+1}: {chan[ll]}x{dim[ll][0]}x{dim[ll][1]} '
                       f'{"(CHW/big data)" if big_data[ll] else "(HWC/little data)"}, ')
            if pool[ll] > 0:
                apb.output(f'{pool[ll]}x{pool[ll]} {"avg" if pool_average[ll] else "max"} '
                           f'pool with stride {pool_stride[ll]}')
            else:
                apb.output(f'no pooling')
            apb.output(f', {kernel_size[ll][0]}x{kernel_size[ll][1]} convolution '
                       f'with stride {stride[ll]} '
                       f'pad {padding[ll]}, {chan[ll+1]}x{dim[ll+1][0]}x{dim[ll+1][1]} out\n')

        apb.output('\n')
        apb.header()

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
            if chan[ll] > tc.MAX_CHANNELS:
                print(f'Layer {ll} is configured for {chan[ll]} inputs, which exceeds '
                      f'the system maximum of {tc.MAX_CHANNELS}.')
                sys.exit(1)
            this_map = []
            for group in range(tc.P_NUMGROUPS):
                if (processor_map[ll] >> group*tc.P_NUMPRO) % 2**tc.P_NUMPRO:
                    this_map.append(group)
            group_map.append(this_map)

        groups_used = []
        for group in range(tc.P_NUMGROUPS):
            if ((processors_used | processor_map[layers]) >> group*tc.P_NUMPRO) % 2**tc.P_NUMPRO:
                groups_used.append(group)

        if embedded_code:
            # Pre-define data memory loader. Inline later when generating RTL sim.
            load.load(embedded_code, apb, big_data[0], processor_map[0], input_size, chan[0],
                      dim[0], data, padding[0], split=split, debug=debug)
            # Pre-define the kernels and bias values
            kern_offs, kern_len = \
                kernels.load(verbose, embedded_code, apb, layers, kernel, kernel_size,
                             quantization, processor_map, chan, debug)
            bias_offs, bias_group, group_bias_max = \
                kernels.load_bias(verbose, embedded_code, apb, layers, bias,
                                  quantization, group_map, chan, debug)

        apb.load_header()

        # Initialize CNN registers

        if verbose:
            print('\nGlobal registers:')
            print('-----------------')

        # Disable completely unused groups
        for group in range(tc.P_NUMGROUPS):
            if group not in groups_used:
                apb.write_ctl(group, tc.REG_CTL, 0,
                              verbose, comment=f' // Disable group {group}')

        # Configure global control registers for used groups
        for _, group in enumerate(groups_used):
            # Zero out Tornado RAM
            if not embedded_code:
                for p in range(tc.P_NUMPRO):
                    for offs in range(tc.TRAM_SIZE):
                        apb.write_tram(group, p, offs, 0, comment='Zero ')
                apb.output('\n')
            # else:
            #     addr = apb_base + tc.C_GROUP_OFFS*group + C_TRAM_BASE
            #     apb.output(f'  memset((uint32_t *) 0x{addr:08x}, 0, '
            #                f'{tc.TRAM_SIZE * tc.P_NUMPRO * 4}); // Zero TRAM {group}\n')
            #     apb.output('\n')

            # Stop state machine - will be overwritten later
            apb.write_ctl(group, tc.REG_CTL, 0x06,
                          verbose, comment=' // Stop SM')
            # SRAM Control - does not need to be changed
            apb.write_ctl(group, tc.REG_SRAM, 0x40e,
                          verbose, comment=' // SRAM control')
            # Number of layers
            apb.write_ctl(group, tc.REG_LCNT_MAX, layers-1,
                          verbose, comment=' // Layer count')
            apb.output('\n')

        if not embedded_code:
            kern_offs, kern_len = \
                kernels.load(verbose, embedded_code, apb, layers, kernel, kernel_size,
                             quantization, processor_map, chan, debug)
            bias_offs, bias_group, group_bias_max = \
                kernels.load_bias(verbose, embedded_code, apb, layers, bias,
                                  quantization, group_map, chan, debug)
        else:
            apb.output('  load_kernels();\n')
            if max(group_bias_max) > 0:
                apb.output('  load_bias();\n')

        if verbose:
            print('\nGlobal configuration:')
            print('---------------------')
            print(f'Used processors    = {processors_used:016x}')
            print(f'Used groups        = {groups_used}')
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
            if ai85:
                print(f'Kernel dimensions  = {kernel_size}')
                print(f'Kernel bits        = {quantization}')
            print(f'Padding            = {padding}')
            print(f'Group with bias    = {bias_group}')
            print(f'Bias offsets       = {bias_offs}')
            print(f'Output offsets     = {out_offset}')
            print(f'Pooling            = {pool}')
            print(f'Pooling stride     = {pool_stride}')
            print('')

        if verbose:
            print('Layer register configuration:')
            print('-----------------------------')

        # Configure per-layer control registers
        for _, group in enumerate(groups_used):
            for ll in range(layers):
                apb.output(f'\n  // Group {group} layer {ll}\n')

                # Configure row count
                # [7:0] maxcount: lower 8 bits = total of width + pad - 1
                # [9:8] pad: 2 bits pad
                apb.write_lreg(group, ll, tc.LREG_RCNT,
                               (padding[ll] << 8) | (dim[ll][0]-1 + 2*padding[ll]),
                               verbose, comment=' // Rows')

                # Configure column count
                # [7:0] width including padding - 1
                # [9:8] pad count (0 = no pad, 1 = half pad, 2 = full pad)
                apb.write_lreg(group, ll, tc.LREG_CCNT,
                               padding[ll] << 8 | (dim[ll][1]-1 + 2 * padding[ll]),
                               verbose, comment=' // Columns')

                # Configure pooling row count
                apb.write_lreg(group, ll, tc.LREG_PRCNT, max(1, pool[ll]-1),
                               verbose, comment=' // Pooling rows')

                # Configure pooling column count
                apb.write_lreg(group, ll, tc.LREG_PCCNT, max(1, pool[ll]-1),
                               verbose, comment=' // Pooling columns')

                # Configure pooling stride count
                apb.write_lreg(group, ll, tc.LREG_STRIDE, pool_stride[ll]-1,
                               verbose, comment=' // Pooling stride')

                # Configure SRAM write pointer -- write ptr is global
                # Get offset to first available instance of the first used processor of the next
                # layer.
                instance = ffs(processor_map[ll+1]) & ~(tc.P_SHARED-1)
                apb.write_lreg(group, ll, tc.LREG_WPTR_BASE, out_offset[ll] // 4 +
                               ((instance % tc.P_SHARED) * tc.INSTANCE_SIZE |
                                ((instance // tc.P_SHARED) << 12)),
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
                apb.write_lreg(group, ll, tc.LREG_WPTR_OFFS, val,
                               verbose, comment=' // Mask offset count')

                # Configure sram read ptr count -- read ptr is local
                # Source address must match write pointer of previous layer (minus global offset)
                apb.write_lreg(group, ll, tc.LREG_RPTR_BASE,
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
                    for t in range(group_map[ll][0]+1, tc.P_NUMGROUPS):
                        # See if any processors other than this one are operating
                        # and set the cnnsiena bit if true
                        if (processor_map[ll] >> (t * tc.P_NUMPRO)) % 2**tc.P_NUMPRO:
                            sources |= 1 << t
                    val |= sources << 12
                apb.write_lreg(group, ll, tc.LREG_LCTL, val,
                               verbose, comment=' // Layer control')

                # Configure mask count
                # Restriction: Every one of the mask memories will have to start from same offset
                # AI84:
                # [6:0]   Max count (output channels)
                # [7]     RFU
                # [14:8]  Starting address for group of 16
                # [15]    RFU
                # [23:16] Bias pointer starting address
                # [24]    Bias enable
                # [31:25] RFU
                # AI85:
                # [15:0]  Max count (output channels)
                # [31:16] Starting address for group of 16
                val = kern_offs[ll] << tc.MCNT_SAD_OFFS \
                    | (kern_len[ll] << tc.MCNT_MAX_OFFS) - (quantization[ll] if ai85 else 1)
                if not ai85:
                    if group == bias_group[ll]:
                        # Enable bias only for one group
                        val |= 0x1000000 | bias_offs[ll] << 16
                apb.write_lreg(group, ll, tc.LREG_MCNT, val,
                               verbose, comment=' // Mask offset and count')

                # Configure tram pointer max
                if pool[ll] > 0:
                    val = max(0, (dim[ll][1] + pool_stride[ll] - pool[ll]) // pool_stride[ll] +
                              2*padding[ll] - 3)
                else:
                    val = max(0, dim[ll][1] + 2*padding[ll] - 3)
                apb.write_lreg(group, ll, tc.LREG_TPTR, val,
                               verbose, comment=' // TRAM ptr max')

                if ai85:
                    # Compensate for the smaller weights by adjusting the output shift
                    if quantization[ll] == 1:
                        val = (1 << 22) | (3 << 13)  # Shift left 3
                    elif quantization[ll] == 2:
                        val = (2 << 22) | (2 << 13)  # Shift left 2
                    elif quantization[ll] == 4:
                        val = (3 << 22) | (1 << 13)  # Shift left 1
                    else:
                        assert quantization[ll] == 8
                        val = 0  # Do not shift

                    if group == bias_group[ll]:
                        # Enable bias only for one group
                        val |= (1 << 12) | bias_offs[ll]
                    apb.write_lreg(group, ll, tc.LREG_POST, val,
                                   verbose, comment=' // AI85 post processing register')

                # Configure mask and processor enables
                # [15:0]  processor enable
                # [31:16] mask enable
                # When the input data is sourced from 16 independent byte streams, all 16
                # processors and compute elements need to be enabled.  If there were only 4 input
                # channels, 0x000f000f would be correct.
                #
                # Enable at most 16 processors and masks
                bits = (processor_map[ll] >> group*tc.P_NUMPRO) % 2**tc.P_NUMPRO
                apb.write_lreg(group, ll, tc.LREG_ENA, bits << 16 | bits,
                               verbose, comment=' // Mask and processor enables')

            if zero_unused:
                for ll in range(layers, tc.MAX_LAYERS):
                    for reg in range(tc.MAX_LREG+1):
                        if reg == tc.LREG_RFU:  # Register 2 not implemented
                            continue
                        apb.write_lreg(group, ll, reg, 0,
                                       verbose, comment=f' // Zero unused layer {ll} registers')

        # Load data memory
        if embedded_code:
            # Do the actual code generation later
            apb.output('\n  load_input(); // Load data input\n\n')
        else:
            load.load(embedded_code, apb, big_data[0], processor_map[0], input_size, chan[0],
                      dim[0], data, padding[0], split=split, debug=debug)

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
            apb.write_ctl(group, tc.REG_CTL, 0x807 | groups_used[0] << 9,
                          verbose, comment=f' // Enable group {group}')

        # Master control - go
        apb.write_ctl(groups_used[0], tc.REG_CTL, 0x07,
                      verbose, comment=f' // Master enable group {groups_used[0]}')

        apb.load_footer()
        # End of input

    in_map = apb.get_mem()

    if verbose:
        print('')

    # Compute layer-by-layer output and chain results into input
    for ll in range(layers):
        out_buf, out_size = cnn_layer(ll + 1, verbose,
                                      input_size, kernel_size[ll], quantization[ll],
                                      chan[ll+1],
                                      [padding[ll], padding[ll]], dilation[ll],
                                      [stride[ll], stride[ll]],
                                      [pool[ll], pool[ll]],
                                      [pool_stride[ll], pool_stride[ll]],
                                      pool_average[ll],
                                      activate[ll],
                                      kernel[ll].reshape(chan[ll+1], input_size[0],
                                                         kernel_size[ll][0], kernel_size[ll][1]),
                                      bias[ll],
                                      data,
                                      ai85=ai85,
                                      debug=debug_computation)

        # Write .mem file for output or create the C cnn_check() function to verify the output
        out_map = [None] * tc.C_GROUP_OFFS * tc.P_NUMGROUPS
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
                filename = None  # Intermediate output - used for layer overwrite check
            filemode = 'a'

        try:
            if filename:
                memfile = open(os.path.join(base_directory, test_name, filename), mode=filemode)
            else:
                memfile = None
            apb.set_memfile(memfile)

            apb.output(f'// {test_name}\n// Expected output of layer {ll+1}\n')
            apb.verify_header()

            # Start at the instance of the first active output processor/channel
            coffs_start = ffs(processor_map[ll+1]) & ~(tc.P_SHARED-1)
            next_layer_map = processor_map[ll+1] >> coffs_start

            for doffs in range(out_size[1] * out_size[2]):
                row, col = divmod(doffs, out_size[2])
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
                    proc = (coffs % tc.MAX_CHANNELS) & ~(tc.P_SHARED-1)
                    offs = tc.C_SRAM_BASE + out_offset[ll] + \
                        (((proc % tc.P_NUMPRO) * tc.INSTANCE_SIZE |
                          (proc // tc.P_NUMPRO) * tc.C_GROUP_OFFS // 4) +
                         doffs) * 4
                    if not ai85 and pool[ll] == 4 and pool_stride[ll] == 4:
                        offs += (doffs // 4) * 8 + 8

                    # If using single layer, make sure we're not overwriting the input
                    if (not overwrite_ok) and in_map[offs >> 2] is not None:
                        print(f'Layer {ll} output for CHW={c},{row}{col} is overwriting '
                              f'input at location {offs:08x}')
                        if not no_error_stop:
                            sys.exit(1)
                    if out_map[offs >> 2] is not None:
                        print(f'Layer {ll} output for CHW={c},{row},{col} is overwriting '
                              f'itself at location {offs:08x}')
                        if not no_error_stop:
                            sys.exit(1)
                    out_map[offs >> 2] = val
                    apb.verify(offs, val, rv=True)
                    coffs += 4

            apb.verify_footer()
        finally:
            if memfile:
                memfile.close()

        input_size = [out_size[0], out_size[1], out_size[2]]
        data = out_buf.reshape(input_size[0], input_size[1], input_size[2])
        in_map = out_map

    with open(os.path.join(base_directory, test_name, filename), mode=filemode) as memfile:
        apb.set_memfile(memfile)

        if fc_weights:
            data = data.flatten()

            out_buf, out_size = linear_layer(verbose=verbose,
                                             do_activation=False,
                                             data=data, weight=fc_weights[0], bias=fc_bias[0],
                                             debug=debug)

            apb.unload(processor_map[layers], input_size, out_offset[layers-1], pool[layers-1],
                       pool_stride[layers-1])
            apb.fc_layer(fc_weights[0], fc_bias[0])
            apb.fc_verify(out_buf)

        apb.main(fc_weights)

    # Close header files
    if embedded_code:
        sampledata_header.close()
        weight_header.close()

    # Create run_test.sv
    if not embedded_code:
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
    tc.set_device(args.ai85)

    # Load configuration file
    cfg, params = yamlcfg.parse(args.config_file)

    # Load weights and biases. This also configures the network's output channels.
    layers, weights, bias, fc_weights, fc_bias, output_channels = \
        checkpoint.load(args.checkpoint_file, cfg['arch'], args.fc_layer, params['quantization'])

    if layers != len(cfg['layers']):
        print(f"Number of layers in the YAML configuration file ({len(cfg['layers'])}) "
              f"does not match the checkpoint file ({layers}).")
        sys.exit(1)

    if any(p < 0 or p > 4*tc.MEM_SIZE for p in params['output_offset']):
        print('Unsupported value for `out_offset` in YAML configuration.')
        sys.exit(1)

    if not args.ai85:
        if any(q != 8 for q in params['quantization']):
            print('All quantization configuration values must be 8 for AI84.')
            sys.exit(1)

        if any(p & 1 != 0 or p < 0 or p > 4 for p in params['pool']):
            print('Unsupported value for `max_pool`/`avg_pool` for AI84 in YAML configuration.')
            sys.exit(1)

        if any(p == 3 or p < 0 or p > 4 for p in params['pool_stride']):
            print('Unsupported value for `pool_stride` in YAML configuration.')
            sys.exit(1)

    if args.stop_after is not None:
        layers = args.stop_after + 1

    processor_map = params['processor_map']

    if 'output_map' in cfg:
        # Use optional configuration value
        output_map = cfg['output_map']
    else:
        if len(processor_map) > layers:
            # When truncating the layers, use the first unused layer's map (--stop-after is used)
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
    output_offset = params['output_offset'][:layers]
    kernel_size = params['kernel_size'][:layers]
    quantization = params['quantization'][:layers]
    pool = params['pool'][:layers]
    pool_stride = params['pool_stride'][:layers]
    padding = params['padding'][:layers]

    # Derived configuration options
    activate = [bool(x) for x in params['relu']]
    pool_average = [bool(x) for x in params['average']]

    print(f"Configuring data set: {cfg['dataset']}.")
    data = sampledata.get(cfg['dataset'])
    input_size = list(data.shape)

    if not args.cmsis_software_nn:
        tn = create_net(args.prefix, args.verbose,
                        args.debug, args.debug_computation, args.no_error_stop,
                        args.overwrite_ok, args.log, args.apb_base, layers, processor_map,
                        input_size, kernel_size, quantization,
                        output_channels, padding,
                        params['dilation'], params['stride'],
                        pool, pool_stride, pool_average, activate,
                        data, weights, bias, params['big_data'], output_map, fc_weights, fc_bias,
                        args.input_split, args.input_offset, output_offset,
                        args.input_filename, args.output_filename, args.c_filename,
                        args.test_dir, args.runtest_filename, args.log_filename,
                        args.zero_unused, args.timeout, not args.top_level, args.verify_writes,
                        args.embedded_code, args.weight_filename, args.sample_filename,
                        args.ai85)
        if not args.embedded_code:
            rtlsim.append_regression(args.top_level, tn, args.queue_name, args.autogen)
    else:
        cmsisnn.create_net(args.prefix, args.verbose, args.debug, args.log,
                           layers, input_size, kernel_size, quantization,
                           output_channels, padding,
                           params['dilation'], params['stride'],
                           pool, pool_stride, pool_average, activate,
                           data, weights, bias, fc_weights, fc_bias,
                           args.c_filename,
                           args.test_dir, args.log_filename,
                           args.weight_filename, args.sample_filename,
                           args.ai85)

    print("SUMMARY OF OPS")
    stats.print_summary()


def signal_handler(_signal, _frame):
    """
    Ctrl+C handler
    """
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()
