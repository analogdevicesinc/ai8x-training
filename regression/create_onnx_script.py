###################################################################################################
#
# Copyright (C) 2023 Analog Devices, Inc. All Rights Reserved.
# This software is proprietary and confidential to Analog Devices, Inc. and its licensors.
#
###################################################################################################
"""
Create onnx bash scripts for test
"""
import argparse
import datetime
import os
import subprocess
import sys

import yaml


def joining(lst):
    """
    Join list based on the ' ' delimiter
    """
    joined_str = ' '.join(lst)
    return joined_str


def time_stamp():
    """
    Take time stamp as string
    """
    time = str(datetime.datetime.now())
    time = time.replace(' ', '.')
    time = time.replace(':', '.')
    return time


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

if not config["Onnx_Status"]:
    sys.exit(1)

folder_path = pathconfig["folder_path"]
output_file_path = pathconfig["output_file_path_onnx"]
train_path = pathconfig["train_path"]

logs_list = os.path.join(folder_path, sorted(os.listdir(folder_path))[-1])

models = []
datasets = []
devices = []
model_paths = []
bias = []
tar_names = []


with open(output_file_path, "w", encoding='utf-8') as onnx_scripts:
    with open(train_path, "r", encoding='utf-8') as input_file:
        contents = input_file.read()
    lines = contents.split("#!/bin/sh ")
    lines = lines[1:]
    contents_t = contents.split()

    j = [i+1 for i in range(len(contents_t)) if contents_t[i] == '--model']
    for index in j:
        models.append(contents_t[index])

    j = [i+1 for i in range(len(contents_t)) if contents_t[i] == '--dataset']
    for index in j:
        datasets.append(contents_t[index])

    j = [i+1 for i in range(len(contents_t)) if contents_t[i] == '--device']
    for index in j:
        devices.append(contents_t[index])

    for i, line in enumerate(lines):
        if "--use-bias" in line:
            bias.append("--use-bias")
        else:
            bias.append("")

    for file_p in sorted(os.listdir(logs_list)):
        temp_path = os.path.join(logs_list, file_p)
        for temp_file in sorted(os.listdir(temp_path)):
            if temp_file.endswith("_checkpoint.pth.tar"):
                temp = os.path.join(temp_path, temp_file)
                model_paths.append(temp)
                tar_names.append(temp_file)

    for i, (model, dataset, bias_value, device_name) in enumerate(
        zip(models, datasets, bias, devices)
    ):
        for tar in model_paths:
            element = tar.split('-')
            modelsearch = element[-4][3:]
            datasearch = element[-3].split('_')[0]
            if datasearch == dataset.split('_')[0] and modelsearch == model:
                tar_path = tar
                timestamp = time_stamp()
                temp = (
                    f"python train.py "
                    f"--model {model} "
                    f"--dataset {dataset} "
                    f"--evaluate "
                    f"--exp-load-weights-from {tar_path} "
                    f"--device {device_name} "
                    f"--summary onnx "
                    f"--summary-filename {model}_{dataset}_{timestamp}_onnx "
                    f"{bias_value}\n"
                )
                onnx_scripts.write(temp)
cmd_command = "bash " + output_file_path

subprocess.run(cmd_command, shell=True, check=True)
