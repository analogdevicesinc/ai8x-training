###################################################################################################
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
###################################################################################################
"""
Unload AI84 HWC memory into standard representation.
"""
from tornadocnn import P_NUMPRO, MAX_CHANNELS, INSTANCE_SIZE, GROUP_SIZE, P_SHARED, C_SRAM_BASE
from utils import ffs, popcount


def unload(memfile, apb_base, processor_map, input_shape, out_offset):
    """
    Unload HWC memory from AI84, writing C code to the `memfile` handle.
    The generated C code is specific to the network configuration passed in in `processor_map`,
    and `input_shape`. Additionally, the generated addresses are offset by `apb_base` and
    `out_offset`. The C code function takes a pointer to a memory array, and the depth of
    the array does not matter (flattened or not flattened) as long as the size is correct.
    """
    memfile.write(f'// Input shape: {input_shape}\n'
                  'void unload(uint8_t *out_buf)\n'
                  '{\n  uint32_t val, *addr, offs;\n\n')

    coffs = ffs(processor_map) & ~(P_SHARED-1)
    next_layer_map = processor_map >> coffs
    read_addr = None
    write_addr = None
    c = 0
    while c < input_shape[2]:
        for doffs in range(input_shape[0] * input_shape[1]):
            row, col = divmod(doffs, input_shape[1])
            this_map = next_layer_map
            this_c = c

            # Get four bytes from memory array
            proc = (coffs % MAX_CHANNELS) & ~(P_SHARED-1)
            offs = out_offset + \
                (((proc % P_NUMPRO) * INSTANCE_SIZE |
                  (proc // P_NUMPRO) * GROUP_SIZE) +
                 doffs) * 4

            if offs != read_addr:
                memfile.write(f'  addr = (uint32_t *) 0x{apb_base + C_SRAM_BASE + offs:08x};\n')
            memfile.write(f'  val = *addr++;\n')
            read_addr = offs + 4

            # Singulate bytes, ignoring unused processors
            for shift in range(4):
                addr = this_c * input_shape[0] * input_shape[1] + row * input_shape[0] + col
                if shift == 0:
                    if addr != write_addr:
                        memfile.write(f'  offs = 0x{addr:04x};\n')
                    else:
                        memfile.write(f'  offs++;\n')
                    write_addr = addr + 1
                if this_map & 1:
                    memfile.write('  out_buf[offs')
                    if shift > 0:
                        memfile.write(f'+0x{0x10 << (shift-1):02x}')
                    memfile.write('] = ')
                    if shift == 0:
                        memfile.write('val')
                    else:
                        memfile.write(f'(val >> {shift * 8})')
                    memfile.write(' & 0xff;\n')
                    this_c += 1
                this_map >>= 1

        coffs += 4
        c += popcount(next_layer_map & 0x0f)
        next_layer_map >>= 4

    memfile.write('}\n')
