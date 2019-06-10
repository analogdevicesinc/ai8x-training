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
from tornadocnn import C_CNN_BASE


def header(memfile, apb_base):
    """
    Write include files and forward definitions to .c file handle `memfile`.
    The APB base address is passed in `apb_base`.
    """
    memfile.write('#include "global_functions.h" // For RTL Simulation\n')
    memfile.write('#include <stdlib.h>\n')
    memfile.write('#include <stdint.h>\n')
    memfile.write('#include <string.h>\n')
    memfile.write('#include "tornadocnn.h"\n\n')

    memfile.write('void cnn_wait(void)\n{\n')
    memfile.write(f'  while ((*((volatile uint32_t *) 0x{apb_base + C_CNN_BASE:08x}) '
                  '& (1<<12)) != 1<<12) ;\n  CNN_COMPLETE; // Signal processing complete\n}\n\n')


def load_header(memfile):
    """
    Write the header for the CNN configuration loader function to `memfile`.
    """
    memfile.write('int cnn_load(void)\n{\n')


def load_footer(memfile):
    """
    Write the footer for the CNN configuration loader function to `memfile`.
    """
    memfile.write('\n  CNN_START; // Allow capture of processing time\n\n  return 1;\n}\n\n')


def main(memfile, classification_layer=False):
    """
    Write the main function (including an optional call to the fully connected layer if
    `fc_layer` is `True`) to `memfile`.
    """
    memfile.write('int main(void)\n{\n  icache_enable();\n')
    memfile.write('  MXC_GCR->perckcn1 &= ~0x20; // Enable AI clock\n')
    memfile.write('  if (!cnn_load()) { fail(); pass(); return 0; }\n  cnn_wait();\n')
    memfile.write('  if (!cnn_check()) fail();\n')
    if classification_layer:
        memfile.write('  if (!fc_layer()) fail();\n')
        memfile.write('  if (!fc_verify()) fail();\n')
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
    memfile.write('  return rv;\n}\n\n')


def fc_layer(memfile, weights, bias):
    """
    Write the call to the fully connected layer with the given `weights` and
    `bias` to `memfile`. The `bias` argument can be `None`.
    """
    memfile.write('// Classification layer (fully connected):\n')
    memfile.write(f'#define FC_IN {weights.shape[1]}\n')
    memfile.write(f'#define FC_OUT {weights.shape[0]}\n')
    memfile.write('#define FC_WEIGHTS {')
    weights.tofile(memfile, sep=',', format='%d')
    memfile.write('}\nstatic const q7_t fc_weights[FC_OUT * FC_IN] = FC_WEIGHTS;\n\n')

    memfile.write('static uint8_t conv_data[FC_IN];\n')
    memfile.write('static q15_t fc_buffer[FC_IN];\n')
    memfile.write('static q15_t fc_output[FC_OUT];\n')
    memfile.write('static q15_t fc_softmax[FC_OUT];\n\n')

    if bias is not None:
        memfile.write('#define FC_BIAS {')
        bias.tofile(memfile, sep=',', format='%d')
        memfile.write('}\nstatic const q7_t fc_bias[FC_OUT] = FC_BIAS;\n\n')

    memfile.write('int fc_layer(void)\n'
                  '{\n  unload(conv_data);\n')

    memfile.write('  arm_fully_connected_q7_q15((q7_t *) conv_data, fc_weights, '
                  'FC_IN, FC_OUT, 0, 7, '
                  f'{"fc_bias" if bias is not None else "NULL"}, '
                  'fc_output, fc_buffer);\n')
    memfile.write('  arm_softmax_q15(fc_output, FC_OUT, fc_softmax);\n\n')

    memfile.write('  return 1;\n}\n\n')


def fc_verify(memfile, data):
    """
    Write the code to verify the fully connected layer to `memfile` against `data`.
    """
    memfile.write('// Expected output of classification layer:\n')
    memfile.write('#define FC_EXPECTED {')
    data.tofile(memfile, sep=',', format='%d')
    memfile.write('}\nstatic q15_t fc_expected[FC_OUT] = FC_EXPECTED;\n\n')
    memfile.write('int fc_verify(void)\n'
                  '{\n')
    memfile.write('  return memcmp(fc_output, fc_expected, FC_OUT * sizeof(q15_t)) == 0;\n}\n\n')
