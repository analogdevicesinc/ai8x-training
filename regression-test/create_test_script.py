###################################################################################################
#
# Copyright (C) 2020-2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
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
args = parser.parse_args()
yaml_path = args.testconf

# Open the YAML file
with open(yaml_path, 'r', encoding='utf-8') as file:
    # Load the YAML content into a Python dictionary
    config = yaml.safe_load(file)

# Folder containing the files to be concatenated
script_path = r"./scripts"

# Output file name and path
output_file_path = r"./scripts/output_file.sh"

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

                temp[i+1] = str(config[log_data][log_model]["epoch"])

                if '--deterministic' not in temp:
                    temp.insert(-1, '--deterministic')

                temp.insert(-1, '--name ' + log_name)

                data_name = temp[k+1]
                if data_name in config and "datapath" in config[data_name]:
                    path_data = config[log_data]["datapath"]
                    temp.insert(-1, '--data ' + path_data)

                temp.append("\n")
                contents = joining(temp)
                output_file.write(contents)
