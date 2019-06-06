###################################################################################################
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
###################################################################################################
"""
Tornado CNN hardware constants - AI84
"""

# AI84
APB_BASE = 0x50100000

# CNN hardware parameters
MAX_LAYERS = 32
TRAM_SIZE = 256
BIAS_SIZE = 256
MASK_WIDTH = 128
C_CNN = 4
C_CNN_BASE = 0
C_TRAM_BASE = C_CNN_BASE + 0x800
C_MRAM_BASE = C_CNN_BASE + 0x4800
C_BRAM_BASE = C_CNN_BASE + 0xC800
C_SRAM_BASE = C_CNN_BASE + 0x10000
C_GROUP_OFFS = 0x100000
P_NUMGROUPS = 4
P_NUMPRO = 16  # Processors per group
P_SHARED = 4  # Processors sharing a data memory

INSTANCE_SIZE = 1024  # x32
GROUP_SIZE = 0x40000
MEM_SIZE = INSTANCE_SIZE * P_NUMPRO * P_NUMGROUPS // P_SHARED  # x32
MAX_CHANNELS = P_NUMPRO * P_NUMGROUPS

# Global registers
REG_CTL = 0
REG_SRAM = 1
REG_LCNT_MAX = 2

# Per-layer registers
LREG_RCNT = 0
LREG_CCNT = 1
LREG_RFU = 2
LREG_PRCNT = 3
LREG_PCCNT = 4
LREG_STRIDE = 5
LREG_WPTR_BASE = 6
LREG_WPTR_OFFS = 7
LREG_RPTR_BASE = 8
LREG_LCTL = 9
LREG_MCNT = 10
LREG_TPTR = 11
LREG_ENA = 12
MAX_LREG = LREG_ENA

# Implementation specifics
BIAS_DIV = 1


def set_device(ai85):
    """
    Change implementation configuration to match the AI84 or AI85, depending on the `ai85`
    bool input.
    """
    global BIAS_DIV  # pylint: disable=global-statement

    print(f'Configuring device: {"AI85" if ai85 else "AI84"}.')
    if not ai85:
        return

    BIAS_DIV = 128
