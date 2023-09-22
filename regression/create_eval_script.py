###################################################################################################
#
# Copyright (C) 2023 Analog Devices, Inc. All Rights Reserved.
# This software is proprietary to Analog Devices, Inc. and its licensors.
#
###################################################################################################
"""
Create training bash scripts for test
"""
import argparse
import os

import yaml


def joining(lst):
    """
    Join list based on the ' ' delimiter
    """
    join_str = ' '.join(lst)
    return join_str


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

# Folder containing the files to be concatenated
script_path = pathconfig["script_path"]

# Output file name and path
output_file_path = pathconfig["output_file_path_evaluation"]

# Loop through all files in the folder
with open(output_file_path, "w", encoding='utf-8') as evaluate_file:
    for filename in os.listdir(script_path):
        # Check if the file is a text file
        if filename.startswith("evaluate"):
            # Open the file and read its contents
            with open(os.path.join(script_path, filename), encoding='utf-8') as input_file:
                contents = input_file.read()
                temp = contents.split()
                temp.insert(1, "\n")

                i = temp.index("--exp-load-weights-from")
                temp[i+1] = temp[i+1][1:]

                temp.insert(-1, "--name " + filename[9:-3])
                temp.insert(-1, "--data /data_ssd")

                temp.append(" \n")
                contents = joining(temp)
                evaluate_file.write(contents)
