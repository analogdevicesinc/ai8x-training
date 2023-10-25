###################################################################################################
#
# Copyright (C) 2023 Analog Devices, Inc. All Rights Reserved.
# This software is proprietary and confidential to Analog Devices, Inc. and its licensors.
#
###################################################################################################
"""
Create the last developed code logs for base testing source
"""
import argparse
import datetime
import os
import subprocess

import requests
import yaml


def joining(lst):
    """
    Join based on the ' ' delimiter
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
script_path = pathconfig["script_path_dev"]
# Output file name and path
output_file_path = pathconfig["output_file_path_dev"]

# global log_file_names
log_file_names = []


def dev_scripts(script_pth, output_file_pth):
    """
    Create training scripts for the last developed code
    """
    with open(output_file_pth, "w", encoding='utf-8') as output_file:
        for filename in os.listdir(script_pth):
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
                    if log_data == "ai85tinierssdface":
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


def dev_checkout():
    """
    Checkout the last developed code
    """
    repository = 'MaximIntegratedAI/ai8x-training'
    branch = 'develop'

    # Send a GET request to the GitHub API to retrieve the commit data
    url = f'https://api.github.com/repos/{repository}/commits?sha={branch}'
    response = requests.get(url)

    if response.status_code == 200:
        commit_hash = response.json()[0]['sha']
        print(f"The hash of the last commit in the '{branch}' branch is: {commit_hash}")
    else:
        print(f"Failed to retrieve commit information. Status code: {response.status_code}")

    commit_num_path = pathconfig['commit_num_path']

    try:
        with open(commit_num_path, "r", encoding='utf-8') as file:
            saved_commit_hash = file.read().strip()
    except FileNotFoundError:
        saved_commit_hash = ""

    if commit_hash != saved_commit_hash:
        with open(commit_num_path, "w", encoding='utf-8') as file:
            file.write(commit_hash)
            dev_scripts(script_path, output_file_path)
            cmd_command = "bash " + output_file_path
            subprocess.run(cmd_command, shell=True, check=True)

            source_path = pathconfig["source_path"]
            destination_path = os.path.join(
                pathconfig["destination_path"],
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
            subprocess.run(['/bin/mv', source_path, destination_path], check=True)


dev_checkout()
