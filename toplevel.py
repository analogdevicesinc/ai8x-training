###################################################################################################
# Copyright (C) 2018-2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
###################################################################################################
"""
Toplevel C file structure generation
"""
import tornadocnn


def header(memfile, apb_base):
    """
    Write include files and forward definitions to .c file handle `memfile`.
    The APB base address is passed in `apb_base`.
    """
    memfile.write('#include "global_functions.h"\n')
    memfile.write('#include <stdlib.h>\n')
    memfile.write('#include <stdint.h>\n')
    memfile.write('#include <string.h>\n')
    memfile.write('\nint cnn_check(void);\n\n')

    memfile.write('void cnn_wait(void)\n{\n')
    memfile.write(f'  while ((*((volatile uint32_t *) 0x{apb_base + tornadocnn.C_CNN_BASE:08x}) '
                  '& (1<<12)) != 1<<12) ;\n}\n\n')


def load_header(memfile):
    """
    Write the header for the CNN configuration loader function to `memfile`.
    """
    memfile.write('int cnn_load(void)\n{\n')


def load_footer(memfile):
    """
    Write the footer for the CNN configuration loader function to `memfile`.
    """
    memfile.write('  return 1;\n}\n\n')


def main(memfile):
    """
    Write the main function to `memfile`.
    """
    memfile.write('int main(void)\n{\n  icache_enable();\n')
    memfile.write('  MXC_GCR->perckcn1 &= ~0x20; // Enable AI clock\n')
    memfile.write('  if (!cnn_load()) { fail(); pass(); return 0; }\n  cnn_wait();\n')
    memfile.write('  if (!cnn_check()) fail();\n')
    memfile.write('  pass();\n  return 0;\n}\n\n')


def verify_header(memfile):
    """
    Write the header for the CNN verification function to `memfile`.
    """
    memfile.write('int cnn_check(void)\n{\n  int rv = 1;\n')


def verify_footer(memfile):
    """
    Write the footer for the CNN verification function to `memfile`.
    """
    memfile.write('  return rv;\n}\n')
