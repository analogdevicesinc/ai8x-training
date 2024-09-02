###################################################################################################
#
# Copyright (C) 2020-2024 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Parses YAML file used to define Quantization Aware Training
"""

import yaml


def parse(yaml_file, msglogger=None):
    """
    Parses `yaml_file` that defines the QAT policy
    """
    policy = None
    with open(yaml_file, mode='r', encoding='utf-8') as stream:
        try:
            policy = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if msglogger is not None:
        msglogger.info(policy)

    if policy and 'start_epoch' not in policy:
        assert False, '`start_epoch` must be defined in QAT policy'
    if policy and 'weight_bits' not in policy:
        assert False, '`weight_bits` must be defined in QAT policy'
    if policy and 'outlier_removal_z_score' not in policy:
        policy['outlier_removal_z_score'] = 8.0
        if msglogger is not None:
            msglogger.info('`outlier_removal_z_score` not defined in QAT policy.'
                           'Using default value of 8.0')

    return policy
