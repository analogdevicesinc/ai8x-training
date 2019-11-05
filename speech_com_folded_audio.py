###################################################################################################
#
# Copyright (C) 2018-2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
###################################################################################################
"""
Classes and functions used to utilize Folded 1D Speech Commands dataset.
"""
import os
import torch
from torch.utils import data
import numpy as np


class SpeechComFolded1D(data.Dataset):
    class_dict = {'backward': 0, 'bed': 1, 'bird': 2, 'cat': 3, 'dog': 4, 'down': 5,
                  'eight': 6, 'five': 7, 'follow': 8, 'forward': 9, 'four': 10, 'go': 11,
                  'happy': 12, 'house': 13, 'learn': 14, 'left': 15, 'marvin': 16, 'nine': 17,
                  'no': 18, 'off': 19, 'on': 20, 'one': 21, 'right': 22, 'seven': 23,
                  'sheila': 24, 'six': 25, 'stop': 26, 'three': 27, 'tree': 28, 'two': 29,
                  'up': 30, 'visual': 31, 'wow': 32, 'yes': 33, 'zero': 34}

    def __init__(self, root, classes, d_type, t_type, transform=None):
        self.classes = classes
        self.d_type = d_type
        self.t_type = t_type
        self.transform = transform
        self.data_file = os.path.join(root, 'SpeechComFolded1D', 'dataset.pt')

        if self.t_type == 'mfcc':
            self.data, self.targets, _, self.data_type = torch.load(self.data_file)
        elif self.t_type == 'keyword':
            self.data, _, self.targets, self.data_type = torch.load(self.data_file)
        else:
            print('Unknown target type: %s' % t_type)
            return

        self.__filter_dtype()

        if self.t_type == 'keyword':
            self.__filter_classes()

    def __filter_dtype(self):
        if self.d_type == 'train':
            idx_to_select = (self.data_type == 0)[:, -1]
        elif self.d_type == 'val':
            idx_to_select = (self.data_type == 1)[:, -1]
        elif self.d_type == 'test':
            idx_to_select = (self.data_type == 2)[:, -1]
        else:
            print('Unknown data type: %s' % self.d_type)
            return

        self.data = self.data[idx_to_select, :]
        self.targets = self.targets[idx_to_select, :]
        del self.data_type

    def __filter_classes(self):
        print('\n')
        initial_new_class_label = len(self.class_dict)
        new_class_label = initial_new_class_label
        for c in self.classes:
            if c not in self.class_dict.keys():
                print('Class is not in the data: %s' % c)
                return
            else:
                print('Class %s, %d' % (c, self.class_dict[c]))
                num_elems = (self.targets == self.class_dict[c]).cpu().sum()
                print('Number of elements in class %s: %d' % (c, num_elems))
                self.targets[(self.targets == self.class_dict[c])] = new_class_label
                new_class_label += 1

        num_elems = (self.targets < initial_new_class_label).cpu().sum()
        print('Number of elements in class unknown: %d' % (num_elems))
        self.targets[(self.targets < initial_new_class_label)] = new_class_label
        self.targets -= initial_new_class_label
        print(np.unique(self.targets.data.cpu()))
        print('\n')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inp, target = self.data[index].type(torch.FloatTensor), int(self.targets[index])

        inp /= 256

        if self.transform is not None:
            inp = self.transform(inp)

        return inp, target
