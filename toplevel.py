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
from armx4weights import convert_to_x4_q7_weights
from tornadocnn import C_CNN_BASE


COPYRIGHT = \
    '// ---------------------------------------------------------------------------\n' \
    '// Copyright (C) 2019 Maxim Integrated Products, Inc.\n' \
    '// All rights reserved. Product of the U.S.A.\n' \
    '// ---------------------------------------------------------------------------\n\n'


def copyright_header(memfile):
    """
    Write the copyright header to .c file handle `memfile`.
    """
    memfile.write(COPYRIGHT)


def header(memfile, apb_base, embedded_code=False, cmsis_nn=False):
    """
    Write include files and forward definitions to .c file handle `memfile`.
    The APB base address is passed in `apb_base`.
    """
    memfile.write('#include <stdlib.h>\n')
    memfile.write('#include <stdint.h>\n')
    if embedded_code:
        memfile.write('#include <string.h>\n')
        memfile.write('#include <stdio.h>\n')
    if not cmsis_nn:
        memfile.write('#include "global_functions.h" // For RTL Simulation\n')
        if embedded_code:
            memfile.write('#include "tmr_utils.h"\n')
    if embedded_code:
        memfile.write('#include "tornadocnn.h"\n')
        memfile.write('#include "weights.h"\n')
        memfile.write('#include "sampledata.h"\n\n')

    if not cmsis_nn:
        if embedded_code:
            memfile.write('uint32_t cnn_time; // Stopwatch\n\n')

        memfile.write('void cnn_wait(void)\n{\n')
        memfile.write(f'  while ((*((volatile uint32_t *) 0x{apb_base + C_CNN_BASE:08x}) '
                      '& (1<<12)) != 1<<12) ;\n')
        if embedded_code:
            memfile.write('  CNN_COMPLETE; // Signal that processing is complete\n')
            memfile.write('  cnn_time = TMR_SW_Stop(MXC_TMR0);\n')
        memfile.write('}\n\n')


def load_header(memfile):
    """
    Write the header for the CNN configuration loader function to `memfile`.
    """
    memfile.write('int cnn_load(void)\n{\n')


def load_footer(memfile, embedded_code=False):
    """
    Write the footer for the CNN configuration loader function to `memfile`.
    """
    if embedded_code:
        memfile.write('\n  CNN_START; // Allow capture of processing time\n')
    memfile.write('\n  return 1;\n}\n\n')


def main(memfile, classification_layer=False, embedded_code=False):
    """
    Write the main function (including an optional call to the fully connected layer if
    `fc_layer` is `True`) to `memfile`.
    """
    memfile.write('int main(void)\n{\n')
    if embedded_code and classification_layer:
        memfile.write('  int i;\n\n')
    memfile.write('  icache_enable();\n\n')
    if embedded_code:
        memfile.write('  SYS_ClockEnable(SYS_PERIPH_CLOCK_AI);\n\n')
    else:
        memfile.write('  MXC_GCR->perckcn1 &= ~0x20; // Enable AI clock\n\n')
    memfile.write('  if (!cnn_load()) { fail(); pass(); return 0; }\n')
    if embedded_code:
        memfile.write('  TMR_SW_Start(MXC_TMR0, NULL);\n')
    memfile.write('  cnn_wait();\n\n')
    memfile.write('  if (!cnn_check()) fail();\n')
    if classification_layer:
        memfile.write('  if (!fc_layer()) fail();\n')
        memfile.write('  if (!fc_verify()) fail();\n')

    if embedded_code:
        memfile.write('\n  printf("\\n*** PASS ***\\n\\n");\n')
        memfile.write('  printf("Time for CNN: %d us\\n\\n", cnn_time);\n\n')
        if classification_layer:
            memfile.write('  printf("Classification results:\\n");\n'
                          '  for (i = 0; i < FC_OUT; i++) {\n'
                          '    printf("[%6d] -> Class %d: %0.1f%%\\n", fc_output[i], '
                          'i, (double) (100.0 * fc_softmax[i] / 32768.0));\n'
                          '  }\n\n')

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


def fc_layer(memfile, weights_fh, weights, bias, cmsis_nn=False):
    """
    Write the call to the fully connected layer with the given `weights` and
    `bias` to `memfile`. The `bias` argument can be `None`.
    """
    memfile.write('// Classification layer (fully connected):\n')
    memfile.write(f'#define FC_IN {weights.shape[1]}\n')
    memfile.write(f'#define FC_OUT {weights.shape[0]}\n')

    weights = convert_to_x4_q7_weights(weights)

    c_define(weights_fh, weights, 'FC_WEIGHTS', '%d', 16)
    memfile.write('static const q7_t fc_weights[FC_OUT * FC_IN] = FC_WEIGHTS;\n\n')

    if not cmsis_nn:
        memfile.write('static uint8_t conv_data[FC_IN];\n')
    memfile.write('static q15_t fc_buffer[FC_IN];\n')
    memfile.write('static q15_t fc_output[FC_OUT];\n')
    memfile.write('static q15_t fc_softmax[FC_OUT];\n\n')

    if bias is not None:
        c_define(weights_fh, bias, 'FC_BIAS', '%d', 16)
        memfile.write('static const q7_t fc_bias[FC_OUT] = FC_BIAS;\n\n')

    if not cmsis_nn:
        memfile.write('int fc_layer(void)\n'
                      '{\n  unload(conv_data);\n')
    else:
        memfile.write('int fc_layer(q7_t *conv_data)\n'
                      '{\n')

    memfile.write('  arm_fully_connected_q7_q15_opt((q7_t *) conv_data, fc_weights, '
                  'FC_IN, FC_OUT, 0, 7, '
                  f'{"fc_bias" if bias is not None else "NULL"}, '
                  'fc_output, fc_buffer);\n')
    memfile.write('  arm_softmax_q15(fc_output, FC_OUT, fc_softmax);\n\n')

    memfile.write('  return 1;\n}\n\n')


def fc_verify(memfile, sampledata, data):
    """
    Write the code to verify the fully connected layer to `memfile` against `data`.
    """
    memfile.write('// Expected output of classification layer:\n')
    c_define(sampledata, data, 'FC_EXPECTED', '%d', 16)
    memfile.write('static q15_t fc_expected[FC_OUT] = FC_EXPECTED;\n\n')
    memfile.write('int fc_verify(void)\n'
                  '{\n')
    memfile.write('  return memcmp(fc_output, fc_expected, FC_OUT * sizeof(q15_t)) == 0;\n}\n\n')


def c_define(memfile, array, define_name, fmt, columns=8):
    """
    Write a #define to `memfile` for array `array` to `define_name`, using format `fmt` and
    creating a line break after `columns` items each.
    `fmt` can have two parts, separated by '%'. The part before the '%' sign is an optional
    prefix and can be empty, the part after the '%' is a formatting directive, e.g. '%08x'.
    """
    prefix, formatting = fmt.split('%')
    memfile.write(f'#define {define_name} {{ \\\n  ')
    for i, e in enumerate(array):
        memfile.write('{prefix}{item:{format}}'.format(prefix=prefix, item=e, format=formatting))
        if i + 1 < len(array):
            memfile.write(', ')
            if (i + 1) % columns == 0:
                memfile.write('\\\n  ')
    memfile.write(' \\\n}\n')
