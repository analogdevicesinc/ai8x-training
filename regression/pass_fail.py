###################################################################################################
#
# Copyright (C) 2023 Analog Devices, Inc. All Rights Reserved.
# This software is proprietary and confidential to Analog Devices, Inc. and its licensors.
#
###################################################################################################
"""
Check the test results
"""
import argparse
import os
import sys

import yaml
from log_comparison import map_value_list, not_found_model

parser = argparse.ArgumentParser()
parser.add_argument('--testconf', help='Enter the config file for the test', required=True)
parser.add_argument('--testpaths', help='Enter the paths for the test', required=True)
args = parser.parse_args()
yaml_path = args.testconf
test_path = args.testpaths

# Open the YAML file
with open(yaml_path, 'r', encoding='utf-8') as yaml_file:
    # Load the YAML content into a Python dictionary
    config = yaml.safe_load(yaml_file)

with open(test_path, 'r', encoding='utf-8') as path_file:
    # Load the YAML content into a Python dictionary
    pathconfig = yaml.safe_load(path_file)

log_path = pathconfig["log_path"]
log_path = os.path.join(log_path, sorted(os.listdir(log_path))[-1])


def check_top_value(diff_file, threshold, map_value):
    """
    Compare Top1 value with threshold
    """
    if not map_value:
        with open(diff_file, 'r', encoding='utf-8') as f:
            model_name = diff_file.split('/')[-1].split('___')[0]
            # Read all lines in the diff_file
            lines = f.readlines()
            # Extract the last line and convert it to a float
            top1 = lines[-1].split()
            try:
                epoch_num = int(top1[0])
            except ValueError:
                print(f"\033[31m\u2718\033[0m Test failed for {model_name}: "
                      f"Cannot convert {top1[0]} to an epoch number.")
                return False
            top1_diff = float(top1[1])

        if top1_diff < threshold:
            print(f"\033[31m\u2718\033[0m Test failed for {model_name} since"
                  f" Top1 value changed {top1_diff} % at {epoch_num}th epoch.")
            return False
        print(f"\033[32m\u2714\033[0m Test passed for {model_name} since"
              f" Top1 value changed {top1_diff} % at {epoch_num}th epoch.")
        return True

    with open(diff_file, 'r', encoding='utf-8') as f:
        model_name = diff_file.split('/')[-1].split('___')[0]
        # Read all lines in the diff_file
        lines = f.readlines()
        # Extract the last line and convert it to a float
        top1 = lines[-1].split()
        try:
            epoch_num = int(top1[0])
        except ValueError:
            print(f"\033[31m\u2718\033[0m Test failed for {model_name}: "
                  f"Cannot convert {top1[0]} to an epoch number.")
            return False
        top1_diff = float(top1[1])
        # top5_diff = float(top1[2])

    if top1_diff < threshold:
        print(f"\033[31m\u2718\033[0m Test failed for {model_name} since"
              f" mAP value changed {top1_diff} % at {epoch_num}th epoch.")
        return False
    print(f"\033[32m\u2714\033[0m Test passed for {model_name} since"
          f" mAP value changed {top1_diff} % at {epoch_num}th epoch.")
    return True


passing = []
for item in not_found_model:
    print("\033[93m\u26A0\033[0m " + "Warning: " + item)

for logs in sorted(os.listdir(log_path)):
    log_name = (logs.split("___"))[0]
    log_model = log_name.split("-")[0]
    log_data = log_name.split("-")[1]

    if log_data in config and log_model in config[log_data]:
        threshold_temp = float(config[f'{log_data}'][f'{log_model}']['threshold'])
    else:
        threshold_temp = 0
    logs = os.path.join(log_path, str(logs))
    map_val = map_value_list[log_name]
    passing.append(check_top_value(logs, threshold_temp, map_val))

if not all(passing):
    print("\033[31mOne or more tests did not pass. Cancelling github actions.")
    sys.exit(1)
