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
Script to merge YouTubeFaces data samples into more compact file series to effetively use during
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

    num_imgs_per_face = 1

    dataset = {}
    part_no = 0

    embedding_path = os.path.join(data_path, 'temp', 'embeddings.json')
    with open(embedding_path, encoding='utf-8') as file:
        embeddings = json.load(file)

    for i, (subj, val) in enumerate(embeddings.items()):  # pylint: disable=too-many-nested-blocks
        if (i % 200) == 0:
            print(f'{i} of {len(embeddings)}')
            if i > 0:
                save_dataset(dataset, data_path, part_no)
                dataset = {}
                part_no += 1

        if subj not in dataset:
            dataset[subj] = {}
        for video_num, val2 in val.items():
            img_folder = os.path.join(data_path, 'temp', subj, str(video_num))

            if video_num not in dataset[subj]:
                dataset[subj][video_num] = {}

            for img_name, embedding in val2.items():
                for idx in range(num_imgs_per_face):
                    img_name = '_'.join([img_name, str(img_size[1]), str(img_size[2]), str(idx)])
                    img_path = os.path.join(img_folder, '.'.join([img_name, 'npy']))
                    img = np.load(img_path).transpose([2, 0, 1])

                    if img.shape == img_size:
                        if np.min(img) != np.max(img):
                            dataset[subj][video_num][img_name] = {'embedding': embedding,
                                                                  'img': img}

    if dataset:
        save_dataset(dataset, data_path, part_no)


def parse_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description='Merge YouTubeFaces data samples to effectively\
                                                  use during training/testing FaceID model.')
    default_data_path = os.path.abspath(__file__)
    for _ in range(3):
        default_data_path = os.path.dirname(default_data_path)
    default_data_path = os.path.join(default_data_path, 'data', 'YouTubeFaces')
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
