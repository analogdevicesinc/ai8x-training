###################################################################################################
#
# Copyright (C) 2023 Analog Devices, Inc. All Rights Reserved.
# This software is proprietary to Analog Devices, Inc. and its licensors.
#
###################################################################################################
"""
Check the test results
"""
import argparse
import os

import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--testpaths', help='Enter the paths for the test', required=True)
args = parser.parse_args()
test_path = args.testpaths

# Open the YAML file
with open(test_path, 'r', encoding='utf-8') as path_file:
    # Load the YAML content into a Python dictionary
    pathconfig = yaml.safe_load(path_file)

eval_path = pathconfig["eval_path"]
eval_file = os.listdir(eval_path)[-1]
directory_path = os.path.join(eval_path, eval_file)
passed = []
failed = []

for filename in sorted(os.listdir(directory_path)):
    path = os.path.join(directory_path, filename)
    file_path = os.path.join(path, os.listdir(path)[0])
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        if "Loss" in content:
            pass_file = filename.split("___")[0]
            passed.append(f"\033[32m\u2714\033[0m Evaluation test passed for {pass_file}.")
        else:
            fail_file = filename.split("___")[0]
            failed.append(f"\033[31m\u2718\033[0m Evaluation test failed for {fail_file}.")

for filename in failed:
    print(filename)
for filename in passed:
    print(filename)
