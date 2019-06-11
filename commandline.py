###################################################################################################
# Copyright (C) 2018-2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
###################################################################################################
"""
Command line parser for Tornado CNN
"""
import argparse
import tornadocnn


def get_parser():
    """
    Return an argparse parser.
    """

    parser = argparse.ArgumentParser(
        description="AI84 CNN Software Generator")
    parser.add_argument('--ai85', action='store_true',
                        help="enable AI85 features")
    parser.add_argument('--apb-base', type=lambda x: int(x, 0),
                        default=tornadocnn.APB_BASE, metavar='N',
                        help=f"APB base address (default: {tornadocnn.APB_BASE:08x})")
    parser.add_argument('--autogen', default='tests', metavar='S',
                        help="directory location for autogen_list (default: 'tests')")
    parser.add_argument('--c-filename', default='test', metavar='S',
                        help="C file name base (default: 'test' -> 'input.c')")
    parser.add_argument('--c-library', action='store_true',
                        help="use C library functions such as memset()")
    parser.add_argument('-d', '--device-code', action='store_true',
                        help="generate embedded code for device instead of RTL simulation")
    parser.add_argument('-f', '--fc-layer', action='store_true',
                        help="add a fully connected classification layer in software "
                             "(default: false)")
    parser.add_argument('-D', '--debug', action='store_true',
                        help="debug mode (default: false)")
    parser.add_argument('--debug-computation', action='store_true',
                        help="debug computation (default: false)")
    parser.add_argument('--config-file', required=True, metavar='S',
                        help="YAML configuration file containing layer configuration")
    parser.add_argument('--checkpoint-file', required=True, metavar='S',
                        help="checkpoint file containing quantized weights")
    parser.add_argument('--input-filename', default='input', metavar='S',
                        help="input .mem file name base (default: 'input' -> 'input.mem')")
    parser.add_argument('--output-filename', default='output', metavar='S',
                        help="output .mem file name base (default: 'output' -> 'output-X.mem')")
    parser.add_argument('--runtest-filename', default='run_test.sv', metavar='S',
                        help="run test file name (default: 'run_test.sv')")
    parser.add_argument('--log-filename', default='log.txt', metavar='S',
                        help="log file name (default: 'log.txt')")
    parser.add_argument('--no-error-stop', action='store_true',
                        help="do not stop on errors (default: stop)")
    parser.add_argument('--input-offset', type=lambda x: int(x, 0), default=0,
                        metavar='N', choices=range(4*tornadocnn.MEM_SIZE),
                        help="input offset (x8 hex, defaults to 0x0000)")
    parser.add_argument('--overwrite-ok', action='store_true',
                        help="allow output to overwrite input (default: warn/stop)")
    parser.add_argument('--queue-name', default='lowp', metavar='S',
                        help="queue name (default: 'lowp')")
    parser.add_argument('-L', '--log', action='store_true',
                        help="redirect stdout to log file (default: false)")
    parser.add_argument('--input-split', type=int, default=1, metavar='N',
                        choices=range(1, tornadocnn.MAX_CHANNELS+1),
                        help="split input into N portions (default: don't split)")
    parser.add_argument('--stop-after', type=int, metavar='N',
                        help="stop after layer")
    parser.add_argument('--prefix', metavar='S', required=True,
                        help="set test name prefix")
    parser.add_argument('--test-dir', metavar='S', required=True,
                        help="set base directory name for auto-filing .mem files")
    parser.add_argument('--top-level', default=None, metavar='S',
                        help="top level name instead of block mode (default: None)")
    parser.add_argument('--timeout', type=int, metavar='N', default=4,
                        help="set timeout (units of 10ms, default 40ms)")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="verbose output (default: false)")
    parser.add_argument('--verify-writes', action='store_true',
                        help="verify write operations (toplevel only, default: false)")
    parser.add_argument('--zero-unused', action='store_true',
                        help="zero unused registers (default: do not touch)")
    return parser.parse_args()
