###################################################################################################
#
# Copyright (C) 2019-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
AFSK demonstration dataset
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import ai8x

BYTES_PER_SAMPLE = 22
TRAIN_TEST_SPLIT = 0.75


class AFSK(Dataset):
    """
    AI85 Audio frequency-shift keying demodulator dataset.
    """
    train0fn = 'zeros.bit'
    train1fn = 'ones.bit'

    def __init__(self, root, train, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.data = None
        self.avail = None

        # Load processed data
        if (os.path.exists(os.path.join(self.processed_path, self.train0fn)) |
                os.path.exists(os.path.join(self.processed_path, self.train1fn))):
            with open(os.path.join(self.processed_path, self.train0fn), 'rb') as fd:
                zerobits = np.fromfile(fd, dtype=np.uint8)
            with open(os.path.join(self.processed_path, self.train1fn), 'rb') as fd:
                onebits = np.fromfile(fd, dtype=np.uint8)
        else:
            raise RuntimeError('Unable to locate training data')

        # Make available an equal amount from each classification
        numbitper = int(min([len(zerobits), len(onebits)]) / BYTES_PER_SAMPLE)

        # Allocate samples between training and testing
        trbits = int(np.ceil(numbitper * TRAIN_TEST_SPLIT))
        tebits = numbitper - trbits

        # Number of bytes for each dataset from each file
        trbytes = trbits * BYTES_PER_SAMPLE
        tebytes = tebits * BYTES_PER_SAMPLE

        # Concatenate allotted bytes from each file and set number of available samples
        if self.train:
            self.data = np.concatenate((zerobits[:trbytes],
                                        onebits[:trbytes]))
            self.avail = trbits * 2
        else:
            self.data = np.concatenate((zerobits[trbytes:trbytes + tebytes],
                                        onebits[trbytes:trbytes + tebytes]))
            self.avail = tebits * 2

    def __len__(self):
        return self.avail

    def __getitem__(self, idx):
        assert self.data is not None and self.avail is not None

        # Index [0 avail) to byte offset
        offs = idx * BYTES_PER_SAMPLE

        sampl = self.data[offs:offs + BYTES_PER_SAMPLE].astype(np.float64)

        # min-max normalization (rescaling)
        _min = sampl.min()
        _max = sampl.max()
        sampl -= _min
        if _min != _max:
            sampl /= _max - _min

        # 1d array -> 2d tensor
        data = torch.tensor(sampl, dtype=torch.float).unsqueeze(0)

        # apply transform, if any
        if self.transform:
            data = self.transform(data)

        # return tensor and classification
        classification = 0 if idx < int(self.avail / 2) else 1
        return data, classification

    @property
    def raw_path(self):
        """Location of raw data."""
        return os.path.join(self.root, self.__class__.__name__, 'wav')

    @property
    def processed_path(self):
        """Location of processed data."""
        return os.path.join(self.root, self.__class__.__name__, 'bits')


def afsk_get_datasets(data, load_train=True, load_test=True):
    """
    Load AFSK dataset.
    """
    (data_dir, args) = data

    transform = transforms.Compose([
        ai8x.normalize(args=args)
    ])

    if load_train:
        train_dataset = AFSK(root=data_dir, train=True, transform=transform)
    else:
        train_dataset = None

    if load_test:
        test_dataset = AFSK(root=data_dir, train=False, transform=transform)
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'AFSK',
        'input': (1, 22),
        'output': ('zerobit', 'onebit'),
        'loader': afsk_get_datasets
    },
]
