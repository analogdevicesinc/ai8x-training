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
MASK_OFFS_AI84 = 128
MASK_OFFS_AI85 = 128  # 4096 if not stacked
MASK_OFFS = MASK_OFFS_AI84

MCNT_SAD_OFFS_AI84 = 8
MCNT_SAD_OFFS_AI85 = 19
MCNT_SAD_OFFS = MCNT_SAD_OFFS_AI84

MCNT_MAX_OFFS_AI84 = 0
MCNT_MAX_OFFS_AI85 = 3
MCNT_MAX_OFFS = MCNT_MAX_OFFS_AI84

C_CNN = 4
C_CNN_BASE = 0
C_TRAM_BASE = C_CNN_BASE + 0x800
C_MRAM_BASE_AI84 = C_CNN_BASE + 0x4800
C_MRAM_BASE_AI85 = C_CNN_BASE + 0x10000
C_MRAM_BASE = C_MRAM_BASE_AI84
C_BRAM_BASE_AI84 = C_CNN_BASE + 0xC800
C_BRAM_BASE_AI85 = C_CNN_BASE + 0x5000
C_BRAM_BASE = C_BRAM_BASE_AI84
C_SRAM_BASE_AI84 = C_CNN_BASE + 0x10000
C_SRAM_BASE_AI85 = C_CNN_BASE + 0x300000
C_SRAM_BASE = C_SRAM_BASE_AI84

C_GROUP_OFFS_AI84 = 0x100000
C_GROUP_OFFS_AI85 = 0x400000
C_GROUP_OFFS = C_GROUP_OFFS_AI84

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
LREG_POST = 13
MAX_LREG_AI84 = LREG_ENA
MAX_LREG_AI85 = LREG_POST
MAX_LREG = MAX_LREG_AI84

BIAS_DIV_AI84 = 1
BIAS_DIV_AI85 = 128
BIAS_DIV = BIAS_DIV_AI84


def set_device(ai85):
    """
    Change implementation configuration to match the AI84 or AI85, depending on the `ai85`
    bool input.
    """
    global BIAS_DIV  # pylint: disable=global-statement
    global MAX_LREG  # pylint: disable=global-statement
    global MCNT_MAX_OFFS  # pylint: disable=global-statement
    global MCNT_SAD_OFFS  # pylint: disable=global-statement
    global C_BRAM_BASE  # pylint: disable=global-statement
    global C_MRAM_BASE  # pylint: disable=global-statement
    global C_SRAM_BASE  # pylint: disable=global-statement
    global C_GROUP_OFFS  # pylint: disable=global-statement
    global MASK_OFFS  # pylint: disable=global-statement

    print(f'Configuring device: {"AI85" if ai85 else "AI84"}.')
    if not ai85:
        return

    BIAS_DIV = BIAS_DIV_AI85
    MCNT_MAX_OFFS = MCNT_MAX_OFFS_AI85
    MCNT_SAD_OFFS = MCNT_SAD_OFFS_AI85
    MAX_LREG = MAX_LREG_AI85
    C_BRAM_BASE = C_BRAM_BASE_AI85
    C_MRAM_BASE = C_MRAM_BASE_AI85
    C_SRAM_BASE = C_SRAM_BASE_AI85
    C_GROUP_OFFS = C_GROUP_OFFS_AI85
    MASK_OFFS = MASK_OFFS_AI85
