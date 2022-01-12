#!/usr/bin/env python3
###################################################################################################
#
# Copyright (C) 2020-2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Script to merge VGGFace-2 data samples into more compact file series to effetively use during
FaceID model training.
"""

import argparse
import json
import os
import pickle

import numpy as np


def save_dataset(data, merged_data_path, part_no):
    """
    Function to save merged file.
    """
    merged_file_path = os.path.join(merged_data_path, f'whole_set_{part_no:02d}.pkl')
    with open(merged_file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main(data_path):  # pylint: disable=too-many-locals
    """
    Main function to iterate over the data samples to merge.
    """
    img_size = (3, 160, 120)

    subj_list = sorted(os.listdir(os.path.join(data_path, 'temp')))
    part_no = 0
    dataset = {}
    num_empty_subjs = 0

    for i, subj in enumerate(subj_list):
        if subj == 'merged':
            print(f'Folder {subj} skipped')
            continue

        if (i % 250) == 0:
            print(f'{i} of {subj_list}')
            if i > 0:
                save_dataset(dataset, data_path, part_no)
                dataset = {}
                part_no += 1

        if subj not in dataset:
            dataset[subj] = {}

        subj_path = os.path.join(data_path, 'temp', subj)
        if not os.path.isdir(subj_path):
            continue

        if not os.listdir(subj_path):
            print(f'Empty folder: {subj_path}')
            num_empty_subjs += 1
            continue

        embedding_path = os.path.join(subj_path, 'embeddings.json')
        with open(embedding_path, encoding='utf-8') as file:
            embeddings = json.load(file)

        for img_name, emb in embeddings.items():
            img_path = os.path.join(subj_path, img_name)
            img = np.load(img_path).transpose([2, 0, 1])

            if img.shape == img_size:
                if np.min(img) != np.max(img):
                    dataset[subj][img_name] = {'embedding': emb, 'img': img}

    if dataset:
        save_dataset(dataset, data_path, part_no)


def parse_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description='Merge VGGFace-2 data samples to effectively use\
                                                  during training/testing FaceID model.')
    default_data_path = os.path.abspath(__file__)
    for _ in range(3):
        default_data_path = os.path.dirname(default_data_path)
    default_data_path = os.path.join(default_data_path, 'data', 'VGGFace-2')
    parser.add_argument('-p', '--data_path', dest='data_path', type=str,
                        default=default_data_path,
                        help='Folder path to processed data')
    parser.add_argument('--type', dest='data_type', type=str, required=True,
                        help='Data type to generate (train/test)')
    args = parser.parse_args()

    data_path = os.path.join(args.data_path, args.data_type)
    return data_path


if __name__ == "__main__":
    data_folder = parse_args()
    main(data_folder)
