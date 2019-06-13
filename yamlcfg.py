###################################################################################################
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
###################################################################################################
"""
YAML Configuration Routines
"""
import sys
import yaml
from tornadocnn import MEM_SIZE


SUPPORTED_DATASETS = ['mnist', 'fashionmnist', 'cifar-10']


def parse(config_file):
    """
    Configure network parameters from the YAML configuration file `config_file`.
    The function returns both the YAML dictionary as well as a settings dictionary.
    """
    # Load configuration file
    with open(config_file) as cfg_file:
        print(f'Reading {config_file} to configure network...')
        cfg = yaml.load(cfg_file, Loader=yaml.SafeLoader)

    if bool(set(cfg) - set(['dataset', 'layers', 'output_map', 'arch'])):
        print(f'Configuration file {config_file} contains unknown key(s).')
        sys.exit(1)

    if 'layers' not in cfg or 'arch' not in cfg or 'dataset' not in cfg:
        print(f'Configuration file {config_file} does not contain '
              f'`layers`, `arch`, or `dataset`.')
        sys.exit(1)

    if bool(set([cfg['dataset'].lower()]) - set(SUPPORTED_DATASETS)):
        print(f'Configuration file {config_file} contains unknown `dataset`.')
        sys.exit(1)

    padding = []
    pool = []
    pool_stride = []
    output_offset = []
    processor_map = []
    average = []
    relu = []
    big_data = []

    for ll in cfg['layers']:
        if bool(set(ll) - set(['max_pool', 'avg_pool', 'pool_stride', 'out_offset',
                               'activate', 'data_format', 'processors', 'pad'])):
            print(f'Configuration file {config_file} contains unknown key(s) for `layers`.')
            sys.exit(1)

        padding.append(ll['pad'] if 'pad' in ll else 1)
        if 'max_pool' in ll:
            pool.append(ll['max_pool'])
            average.append(0)
        elif 'avg_pool' in ll:
            pool.append(ll['avg_pool'])
            average.append(1)
        else:
            pool.append(0)
            average.append(0)
        pool_stride.append(ll['pool_stride'] if 'pool_stride' in ll else 0)
        output_offset.append(ll['out_offset'] if 'out_offset' in ll else 0)
        if 'processors' not in ll:
            print('`processors` key missing for layer in YAML configuration.')
            sys.exit(1)
        processor_map.append(ll['processors'])
        if 'activate' in ll:
            if ll['activate'].lower() == 'relu':
                relu.append(1)
            else:
                print('Unknown value for `activate` in YAML configuration.')
                sys.exit(1)
        else:
            relu.append(0)
        if 'data_format' in ll:
            if big_data:  # Sequence not empty
                print('`data_format` can only be configured for the first layer.')
                sys.exit(1)

            df = ll['data_format'].lower()
            if df in ['hwc', 'little']:
                big_data.append(False)
            elif df in ['chw', 'big']:
                big_data.append(True)
            else:
                print('Unknown value for `data_format` in YAML configuration.')
                sys.exit(1)
        else:
            big_data.append(False)

    if any(p < 0 or p > 2 for p in padding):
        print('Unsupported value for `pad` in YAML configuration.')
        sys.exit(1)
    if any(p & 1 != 0 or p < 0 or p > 4 for p in pool):
        print('Unsupported value for `max_pool`/`avg_pool` in YAML configuration.')
        sys.exit(1)
    if any(p == 3 or p < 0 or p > 4 for p in pool_stride):
        print('Unsupported value for `pool_stride` in YAML configuration.')
        sys.exit(1)
    if any(p < 0 or p > 4*MEM_SIZE for p in output_offset):
        print('Unsupported value for `out_offset` in YAML configuration.')
        sys.exit(1)

    settings = {}
    settings['padding'] = padding
    settings['pool'] = pool
    settings['pool_stride'] = pool_stride
    settings['output_offset'] = output_offset
    settings['processor_map'] = processor_map
    settings['average'] = average
    settings['relu'] = relu
    settings['big_data'] = big_data

    # We don't support changing the following, but leave as parameters
    settings['dilation'] = [[1, 1]] * len(cfg['layers'])
    settings['kernel_size'] = [[3, 3]] * len(cfg['layers'])
    settings['stride'] = [1] * len(cfg['layers'])

    return cfg, settings
