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
import numpy as np

from tornadocnn import C_SRAM_BASE, C_GROUP_OFFS, P_NUMPRO, P_SHARED, INSTANCE_SIZE, \
    MAX_CHANNELS
from utils import s2u


def load(embedded_code, apb, chw, processor_map, input_size, chan, dim, data, padding,
         split=1, debug=False):
    """
    Create C code to load data input in CHW format (if `chw` is `True`) or HWC format
    for the `processor_map`. Data `data` is organized in `chan` channels, and `dim` dimensions
    and the input size is `input_size`.
    The code performs optional `padding`, can `split` the input into more than one chunk
    and has optional `debug` output.
    The code is target for simulation (`embedded_code` == `False`) or embedded hardware (`True`).
    Output is written to the `apb` object.
    """
    input_list = []

    if not embedded_code:
        apb.output('\n\n  ')
    apb.output(f'// {chan}-channel {dim[0]}x{dim[1]} data input:\n')
    c = 0
    data_offs = 0
    step = 1 if chw else 4
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

        if chw:
            assert split > 0

            # CHW ("Big Data") - Separate channel sequences (BBBBB....GGGGG....RRRRR....)
            if embedded_code and split == 1:
                # Create optimized code when we're not splitting the input
                apb.output(f'// CHW (big data): {dim[0]}x{dim[1]}, channel {c}\n')
                offs = 0
                code_buffer = np.zeros(input_size[1] * input_size[2] // 4, dtype=np.int64)
                addr = data_offs

                val = 0
                for row in range(input_size[1]):
                    for col in range(input_size[2]):
                        shift = (row * input_size[2] + col) % 4
                        val |= (s2u(data[c][row][col]) & 0xff) << (shift * 8)
                        if shift == 3:
                            apb.check_overwrite(data_offs & ~3)
                            code_buffer[offs] = val
                            offs += 1
                            val = 0
                        data_offs += 1

                if shift != 3:
                    apb.check_overwrite(data_offs & ~3)
                    code_buffer[offs] = val
                    offs += 1

                apb.output_define(code_buffer, f'INPUT_{ch}', '0x%08x', 8)
                apb.output('static const uint32_t '
                           f'input_{ch}[{offs}] = INPUT_{ch};\n\n')
                input_list.append((addr, ch, offs))

                apb.data_offs = data_offs  # For mixed HWC/CHW operation
            else:
                if embedded_code:
                    apb.output('void load_input(void)\n{\n')

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
            if not embedded_code:
                apb.output('  ')
            apb.output(f'// HWC (little data): {dim[0]}x{dim[1]}, '
                       f'channels {c} to {min(c+3, chan-1)}\n')

            if embedded_code:
                offs = 0
                code_buffer = np.zeros(input_size[1] * input_size[2], dtype=np.int64)
                addr = data_offs

            for row in range(input_size[1]):
                for col in range(input_size[2]):
                    # Always write multiple of four bytes even for last input
                    val = 0
                    for i in range(chan - c):
                        val |= (s2u(data[c + i][row][col]) & 0xff) << (i * 8)

                    apb.check_overwrite(data_offs)
                    if not embedded_code:
                        apb.write(data_offs, val)
                    else:
                        code_buffer[offs] = val
                        offs += 1
                    apb.data_offs = data_offs  # For mixed HWC/CHW operation
                    data_offs += 4

            if embedded_code:
                apb.output_define(code_buffer, f'INPUT_{ch}', '0x%08x', 8, weights=False)
                apb.output('static const uint32_t '
                           f'input_{ch}[{input_size[1] * input_size[2]}] = INPUT_{ch};\n\n')
                input_list.append((addr, ch, offs))

            c += 4

        apb.write_byte_flush(0)
        if c >= chan:
            # Consumed all available channels
            break

    if embedded_code:
        if input_list:
            apb.output('void load_input(void)\n{\n')
            for _, (addr, ch, offs) in enumerate(input_list):
                apb.output(f'  memcpy((uint32_t *) 0x{apb.apb_base + addr:08x}, input_{ch}, '
                           f'sizeof(uint32_t) * {offs});\n')
        apb.output('}\n\n')

    if not embedded_code:
        apb.output(f'  // End of data input\n\n')
