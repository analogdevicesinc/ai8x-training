###################################################################################################
#
# Copyright (C) 2022-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Parses YAML file used to define Object Detection Parameters YAML File
"""

import yaml


def parse(yaml_file):
    """
    Parses `yaml_file` that defines the Object Detection Parameters
    """
    parameters = None
    with open(yaml_file, mode='r', encoding='utf-8') as stream:
        try:
            parameters = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    print(parameters)
    assert parameters

    try:
        _ = parameters['multi_box_loss']['alpha']
        _ = parameters['multi_box_loss']['neg_pos_ratio']
        _ = parameters['nms']['min_score']
        _ = parameters['nms']['max_overlap']
        _ = parameters['nms']['top_k']
    except KeyError as ke:
        raise KeyError('All parameter fields should be set. ' +
                       'Please see sample .yaml file: obj_detection_params.yaml') from ke
    return parameters
