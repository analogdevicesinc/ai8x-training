###################################################################################################
#
# Copyright (C) 2018-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
###################################################################################################
"""
Classes and functions used to utilize Folded 1D Speech Commands dataset.
"""
import os

import numpy as np
import torch
from torchvision import transforms

import ai8x


class SpeechComFolded1D(torch.utils.data.Dataset):
    """
    `SpeechCom v0.02 <http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz>`
    Dataset, 1D folded.
    """
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


class SpeechComFolded1D_20(SpeechComFolded1D):
    """
    `SpeechCom v0.02 <http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz>`
    Dataset, 1D folded.
    """


def speechcomfolded1D_get_datasets(data, load_train=True, load_test=True, num_classes=6):
    """
    Load the folded 1D version of SpeechCom dataset

    The dataset is loaded from the archive file, so the file is required for this version.

    The dataset originally includes 30 keywords. A dataset is formed with 7 (or 21) classes which includes
    6 (or 20) of the original keywords and the rest of the
    dataset is used to form the last class, i.e class of the others.
    The dataset is split into training, validation and test sets. 80:10:10 training:validation:test
    split is used by default.

    Data is augmented to 3x duplicate data by randomly stretch, shift and randomly add noise where
    the stretching coefficient, shift amount and noise variance are randomly selected between
    0.8 and 1.3, -0.1 and 0.1, 0 and 1, respectively.
    """
    (data_dir, args) = data

    transform = transforms.Compose([
        ai8x.normalize(args=args)
    ])

    if num_classes == 6:
        classes = ['up', 'down', 'left', 'right', 'stop', 'go']
    elif num_classes == 20:
        classes = ['up', 'down', 'left', 'right', 'stop', 'go', 'yes', 'no', 'on', 'off', 'one',
                   'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero']
    else:
        raise ValueError(f'Unsupported num_classes {num_classes}')

    if load_train:
        train_dataset = SpeechComFolded1D(root=data_dir, classes=classes, d_type='train',
                                          transform=transform, t_type='keyword')
    else:
        train_dataset = None

    if load_test:
        test_dataset = SpeechComFolded1D(root=data_dir, classes=classes, d_type='val',
                                         transform=transform, t_type='keyword')

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


def speechcomfolded1D_20_get_datasets(data, load_train=True, load_test=True):
    """
    Load the folded 1D version of SpeechCom dataset for 20 classes

    The dataset is loaded from the archive file, so the file is required for this version.

    The dataset originally includes 30 keywords. A dataset is formed with 21 classes which includes
    20 of the original keywords and the rest of the dataset is used to form the last class, i.e.,
    class of the others.
    The dataset is split into training, validation and test sets. 80:10:10 training:validation:test
    split is used by default.

    Data is augmented to 3x duplicate data by randomly stretch, shift and randomly add noise where
    the stretching coefficient, shift amount and noise variance are randomly selected between
    0.8 and 1.3, -0.1 and 0.1, 0 and 1, respectively.
    """
    return speechcomfolded1D_get_datasets(data, load_train, load_test, num_classes=20)
