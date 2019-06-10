###################################################################################################
# Copyright (C) 2018-2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
###################################################################################################
"""
Load Tornado CNN data memory
"""
import sys

from tornadocnn import C_SRAM_BASE, C_GROUP_OFFS, P_NUMPRO, P_SHARED, INSTANCE_SIZE, \
    MAX_CHANNELS
from utils import s2u


def load(apb, big_data, processor_map, input_size, chan, dim, data, padding, split=0, debug=False):
    """
    Create C code to load data input in the format `big_data` for the `processor_map`. Data `data`
    is `chan` channels, `dim` dimensions and the input size is `input_size`. The code performs
    optional `padding`, `split` input and has optional `debug` output.
    """
    apb.output(f'\n\n  // {chan}-channel data input\n')
    c = 0
    data_offs = 0
    step = 1 if big_data else 4
    for ch in range(0, MAX_CHANNELS, step):
        if not (processor_map >> ch) % 2**step:
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

        if big_data:
            # CHW ("Big Data") - Separate channel sequences (BBBBB....GGGGG....RRRRR....)
            apb.output(f'  // CHW (big data): {dim[0]}x{dim[1]}, channel {c}\n')

            chunk = input_size[1] // split

            # (Note: We do not need to flush here, since that is done at the
            # end of each channel's output below)
            if split > 1:
                # Add top pad
                for _ in range(padding):
                    for _ in range(input_size[2]):
                        apb.write_byte(data_offs, 0)
                        data_offs += 1
            row = 0
            for s in range(split):
                if split > 1 and s + 1 < split:
                    overlap = padding
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
                for _ in range(padding):
                    for _ in range(input_size[2]):
                        apb.write_byte(data_offs, 0)
                        data_offs += 1
            c += 1
        else:
            # HWC ("Little Data") - Four channels packed into a word (0BGR0BGR0BGR0BGR0BGR....)
            apb.output(f'  // HWC (little data): {dim[0]}x{dim[1]}, '
                       f'channels {c} to {min(c+3, chan-1)}\n')

            for row in range(input_size[1]):
                for col in range(input_size[2]):
                    if c < chan:
                        apb.write_byte(data_offs, s2u(data[c][row][col]))
                    else:
                        apb.write_byte(data_offs, 0)
                    data_offs += 1
                    # Always write multiple of four bytes even for last input
                    if c+1 < chan:
                        apb.write_byte(data_offs, s2u(data[c+1][row][col]))
                    else:
                        apb.write_byte(data_offs, 0)
                    data_offs += 1
                    if c+2 < chan:
                        apb.write_byte(data_offs, s2u(data[c+2][row][col]))
                    else:
                        apb.write_byte(data_offs, 0)
                    data_offs += 1
                    if c+3 < chan:
                        apb.write_byte(data_offs, s2u(data[c+3][row][col]))
                    else:
                        apb.write_byte(data_offs, 0)
                    data_offs += 1
            c += 4

        apb.write_byte_flush(0)
        if c >= chan:
            # Consumed all available channels
            break

    apb.output(f'  // End of data input\n\n')
