###################################################################################################
#
# Copyright (C) 2024 Analog Devices, Inc. All Rights Reserved.
# This software is proprietary to Analog Devices, Inc. and its licensors.
#
###################################################################################################
"""
Main classes and functions for background image dataset BG-20k
"""

import os
import sys

from torch.utils.data import Dataset

import cv2


class BG20K(Dataset):
    """
    Dataloader for BG-20k background image dataset.

    BG-20k contains 20,000 high-resolution background images excluded salient objects,
    which can be used to help generate high quality synthetic data [1].

    [1] https://github.com/JizhiziLi/GFM
    """
    def __init__(self, root_dir, d_type, transform=None):
        self.test_to_val_split_ratio = 4
        self.transform = transform
        self.data_list = []
        self.is_truncated = False
        self.root_dir = os.path.join(root_dir, 'BG-20k')

        if d_type == 'train':
            self.data_dir = os.path.join(self.root_dir, 'train')
            self.start_ratio = 0
            self.end_ratio = None
        elif d_type == 'val':
            self.data_dir = os.path.join(self.root_dir, 'testval')
            self.start_ratio = 0
            self.end_ratio = 1. / (1. + self.test_to_val_split_ratio)
        elif d_type == 'test':
            self.data_dir = os.path.join(self.root_dir, 'testval')
            self.start_ratio = 1. / (1. + self.test_to_val_split_ratio)
            self.end_ratio = None
        else:
            print(f'Unknown data type: {d_type}')
            return

        self.__check_dataset()
        self.__gen_data_paths()

    def __check_dataset(self):
        if os.path.exists(self.data_dir):
            if len(os.listdir(self.data_dir)) > 0:
                return True

        print('\nDownload the archive file from: https://github.com/JizhiziLi/GFM to path \
              [data_dir]/BK-20k.')
        print('The download process may require additional authentication.')

        sys.exit()

    def __gen_data_paths(self):
        file_list = sorted(os.listdir(self.data_dir))

        start_idx = int(len(file_list) * self.start_ratio)
        if self.end_ratio is None:
            end_idx = None
        else:
            end_idx = int(len(file_list) * self.end_ratio)

        for f_name in file_list[start_idx:end_idx]:
            self.data_list.append(f_name)

    def __len__(self):
        if self.is_truncated:
            return 1
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.data_list[index])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = 1

        if self.transform is not None:
            image = self.transform(image)

        return image, label
