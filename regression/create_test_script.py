###################################################################################################
#
# Copyright (C) 2023 Analog Devices, Inc. All Rights Reserved.
# This software is proprietary and confidential to Analog Devices, Inc. and its licensors.
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
output_file_path = pathconfig["output_file_path"]

# global log_file_names
log_file_names = []

# Loop through all files in the folder
with open(output_file_path, "w", encoding='utf-8') as output_file:
    for filename in os.listdir(script_path):
        # Check if the file is a text file
        if filename.startswith("train"):
            # Open the file and read its contents
            with open(os.path.join(script_path, filename), encoding='utf-8') as input_file:
                contents = input_file.read()

                temp = contents.split()
                temp.insert(1, "\n")
                i = temp.index('--epochs')
                j = temp.index('--model')
                k = temp.index('--dataset')

                if config["Qat_Test"]:
                    if '--qat-policy' in temp:
                        x = temp.index('--qat-policy')
                        temp[x+1] = "policies/qat_policy.yaml"
                    else:
                        temp.insert(-1, ' --qat-policy policies/qat_policy.yaml')

                log_model = temp[j+1]
                log_data = temp[k+1]

                if log_model == "ai87imageneteffnetv2":
                    num = temp.index("--batch-size")
                    temp[num+1] = "128"

                log_name = temp[j+1] + '-' + temp[k+1]
                log_file_names.append(filename[:-3])

                if log_data == "FaceID":
                    continue

                if log_data == "VGGFace2_FaceDetection":
                    continue

                try:
                    temp[i+1] = str(config[log_data][log_model]["epoch"])
                except KeyError:
                    print(f"\033[93m\u26A0\033[0m Warning: {temp[j+1]} model is" +
                          " missing information in test configuration files.")
                    continue

                if '--deterministic' not in temp:
                    temp.insert(-1, '--deterministic')

                temp.insert(-1, '--name ' + log_name)

                try:
                    path_data = config[log_data]["datapath"]
                    temp[i+1] = str(config[log_data][log_model]["epoch"])
                except KeyError:
                    print(f"\033[93m\u26A0\033[0m Warning: {temp[j+1]} model is" +
                          " missing information in test configuration files.")
                    continue

                temp.insert(-1, '--data ' + path_data)
                temp.append("\n")

                contents = joining(temp)
                output_file.write(contents)
