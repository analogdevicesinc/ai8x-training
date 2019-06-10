#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
###################################################################################################
"""
Test flatten() software operator
"""
import numpy as np
from tornadocnn import P_NUMPRO, MAX_CHANNELS, INSTANCE_SIZE, GROUP_SIZE, P_SHARED, C_SRAM_BASE, \
    APB_BASE
from utils import ffs, popcount


MEM_INVALID = -2**63  # When  encountering this value, we know the array value was not initialized
MEM_SIZE = 0x10000 >> 2


def unload_flatten(apb_base, processor_map, in_array, in_size,
                   chan, out_array, out_offset, out_len):
    """
    Unload and flatten an HWC `in_array` of dimensions from AI84 and return it in `out_array`.
    """
    def get_val(offs):
        """
        Returns value stored at offset `offs` in the memory array.
        """
        if offs >= (MEM_SIZE << 2) or offs < 0:
            raise RuntimeError(f'Offset {offs:04x} is invalid for the memory array.')
        if offs & 3:
            raise RuntimeError(f'Offset {offs:04x} should be a 32-bit address.')
        if in_array[offs >> 2] == MEM_INVALID:
            raise RuntimeError(f'Trying to read from uninitialized memory at location {offs:04x}.')
        return in_array[offs >> 2]

    print('// Custom unload for this network:\n'
          f'// Input length: {in_size}, Output length: {out_len}, Channels: {chan}\n'
          'void unload_flatten(uint8_t *out_buf)\n'
          '{\n  uint32_t val, *addr, offs;\n')

    coffs = ffs(processor_map) & ~(P_SHARED-1)
    next_layer_map = processor_map >> coffs
    read_addr = None
    write_addr = None
    c = 0
    while c < chan:
        for doffs in range(in_size[0] * in_size[1]):
            row, col = divmod(doffs, in_size[1])
            this_map = next_layer_map
            this_c = c

            # Get four bytes from memory array
            proc = (coffs % MAX_CHANNELS) & ~(P_SHARED-1)
            offs = out_offset + \
                (((proc % P_NUMPRO) * INSTANCE_SIZE |
                  (proc // P_NUMPRO) * GROUP_SIZE) +
                 doffs) * 4

            val = get_val(offs)

            if offs != read_addr:
                print(f'  addr = (uint32_t *) 0x{apb_base + C_SRAM_BASE + offs:08x};')
            print(f'  val = *addr++;')
            read_addr = offs + 4

            # Singulate bytes, ignoring unused processors
            for shift in range(4):
                addr = this_c * in_size[0] * in_size[1] + row * in_size[0] + col
                if shift == 0:
                    if addr != write_addr:
                        print(f'  offs = 0x{addr:04x};')
                    else:
                        print(f'  offs++;')
                    write_addr = addr + 1
                if this_map & 1:
                    out_array[addr] = val & 0xff
                    print('  out_buf[offs', end='')
                    if shift > 0:
                        print(f'+0x{0x10 << (shift-1):02x}', end='')
                    print('] = ', end='')
                    if shift == 0:
                        print('val', end='')
                    else:
                        print(f'(val >> {shift * 8})', end='')
                    print(' & 0xff;')
                    this_c += 1
                this_map >>= 1
                val >>= 8

        coffs += 4
        c += popcount(next_layer_map & 0x0f)
        next_layer_map >>= 4

    print('}')


def main():
    """
    main() - when invoked from command line
    """

    # Create memory image
    mem_image = np.full(MEM_SIZE, MEM_INVALID, dtype=np.int64)

    # Fill image with known values
    mem_image[0x0000 >> 2] = 0x00540055
    mem_image[0x0004 >> 2] = 0x007f0070
    mem_image[0x0008 >> 2] = 0x0e530345
    mem_image[0x000c >> 2] = 0x0946084e
    mem_image[0x0010 >> 2] = 0x044d0045
    mem_image[0x0014 >> 2] = 0x00630051
    mem_image[0x0018 >> 2] = 0x005e0c33
    mem_image[0x001c >> 2] = 0x043d2f41
    mem_image[0x0020 >> 2] = 0x0900002d
    mem_image[0x0024 >> 2] = 0x00000018
    mem_image[0x0028 >> 2] = 0x001c0000
    mem_image[0x002c >> 2] = 0x00180814
    mem_image[0x0030 >> 2] = 0x00000022
    mem_image[0x0034 >> 2] = 0x00000200
    mem_image[0x0038 >> 2] = 0x0005000f
    mem_image[0x003c >> 2] = 0x0002001e
    mem_image[0x4000 >> 2] = 0x2d051a0d
    mem_image[0x4004 >> 2] = 0x394e141a
    mem_image[0x4008 >> 2] = 0x2039141b
    mem_image[0x400c >> 2] = 0x0c000029
    mem_image[0x4010 >> 2] = 0x18130913
    mem_image[0x4014 >> 2] = 0x0a6c0000
    mem_image[0x4018 >> 2] = 0x004f0000
    mem_image[0x401c >> 2] = 0x001a0000
    mem_image[0x4020 >> 2] = 0x00000008
    mem_image[0x4024 >> 2] = 0x00500000
    mem_image[0x4028 >> 2] = 0x005a0000
    mem_image[0x402c >> 2] = 0x004a0000
    mem_image[0x4030 >> 2] = 0x0f190b0e
    mem_image[0x4034 >> 2] = 0x225b0c17
    mem_image[0x4038 >> 2] = 0x006b030f
    mem_image[0x403c >> 2] = 0x00570903
    mem_image[0x8000 >> 2] = 0x381e3b00
    mem_image[0x8004 >> 2] = 0x6c233a00
    mem_image[0x8008 >> 2] = 0x6c002500
    mem_image[0x800c >> 2] = 0x2d000000
    mem_image[0x8010 >> 2] = 0x38432800
    mem_image[0x8014 >> 2] = 0x646a1700
    mem_image[0x8018 >> 2] = 0x53680500
    mem_image[0x801c >> 2] = 0x2734063d
    mem_image[0x8020 >> 2] = 0x10573427
    mem_image[0x8024 >> 2] = 0x177f2a50
    mem_image[0x8028 >> 2] = 0x0a5a004b
    mem_image[0x802c >> 2] = 0x0028003c
    mem_image[0x8030 >> 2] = 0x082d0e07
    mem_image[0x8034 >> 2] = 0x00400009
    mem_image[0x8038 >> 2] = 0x0a1a0419
    mem_image[0x803c >> 2] = 0x00170809

    flattened = np.array([
        0x55, 0x70, 0x45, 0x4e, 0x45, 0x51, 0x33, 0x41,
        0x2d, 0x18, 0x00, 0x14, 0x22, 0x00, 0x0f, 0x1e,
        0x00, 0x00, 0x03, 0x08, 0x00, 0x00, 0x0c, 0x2f,
        0x00, 0x00, 0x00, 0x08, 0x00, 0x02, 0x00, 0x00,
        0x54, 0x7f, 0x53, 0x46, 0x4d, 0x63, 0x5e, 0x3d,
        0x00, 0x00, 0x1c, 0x18, 0x00, 0x00, 0x05, 0x02,
        0x00, 0x00, 0x0e, 0x09, 0x04, 0x00, 0x00, 0x04,
        0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x0d, 0x1a, 0x1b, 0x29, 0x13, 0x00, 0x00, 0x00,
        0x08, 0x00, 0x00, 0x00, 0x0e, 0x17, 0x0f, 0x03,
        0x1a, 0x14, 0x14, 0x00, 0x09, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x0b, 0x0c, 0x03, 0x09,
        0x05, 0x4e, 0x39, 0x00, 0x13, 0x6c, 0x4f, 0x1a,
        0x00, 0x50, 0x5a, 0x4a, 0x19, 0x5b, 0x6b, 0x57,
        0x2d, 0x39, 0x20, 0x0c, 0x18, 0x0a, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x0f, 0x22, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3d,
        0x27, 0x50, 0x4b, 0x3c, 0x07, 0x09, 0x19, 0x09,
        0x3b, 0x3a, 0x25, 0x00, 0x28, 0x17, 0x05, 0x06,
        0x34, 0x2a, 0x00, 0x00, 0x0e, 0x00, 0x04, 0x08,
        0x1e, 0x23, 0x00, 0x00, 0x43, 0x6a, 0x68, 0x34,
        0x57, 0x7f, 0x5a, 0x28, 0x2d, 0x40, 0x1a, 0x17,
        0x38, 0x6c, 0x6c, 0x2d, 0x38, 0x64, 0x53, 0x27,
        0x10, 0x17, 0x0a, 0x00, 0x08, 0x00, 0x0a, 0x00,
    ], dtype=np.int64)

    computed = np.empty_like(flattened)
    processor_map = 0x0000000000000fff
    in_size = (4, 4)
    out_chan = 12
    out_offset = 0
    unload_flatten(APB_BASE, processor_map, mem_image, in_size,
                   out_chan, computed, out_offset, len(computed))

    print('\n')
    print("// SUCCESS" if np.array_equal(flattened, computed) else "// *** FAILURE ***")


if __name__ == '__main__':
    main()
