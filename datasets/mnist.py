#
# Copyright (c) 2018 Intel Corporation
# Portions Copyright (C) 2019-2023 Maxim Integrated Products, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
MNIST and FashionMNIST Datasets
"""
import torchvision
from torchvision import transforms

import ai8x


def mnist_get_datasets(data, load_train=True, load_test=True):
    """
    Load the MNIST dataset.

    The original training dataset is split into training and validation sets (code is
    inspired by https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb).
    By default we use a 90:10 (45K:5K) training:validation split.

    The output of torchvision datasets are PIL Image images of range [0, 1].
    We transform them to Tensors of normalized range [-128/128, +127/128]
    https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py

    Data augmentation: 4 pixels are padded on each side, and a 32x32 crop is randomly sampled
    from the padded image or its horizontal flip.
    This is similar to [1] and some other work that use CIFAR10.

    [1] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu. Deeply Supervised Nets.
    arXiv:1409.5185, 2014
    """
    (data_dir, args) = data

    if load_train:
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=5),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True,
                                                   transform=train_transform)
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True,
                                                  transform=test_transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


def fashionmnist_get_datasets(data, load_train=True, load_test=True):
    """
    Load the FashionMNIST dataset.

    The original training dataset is split into training and validation sets (code is
    inspired by https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb).
    By default we use a 90:10 (45K:5K) training:validation split.

    The output of torchvision datasets are PIL Image images of range [0, 1].
    We transform them to Tensors of normalized range [-1, 1]
    https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py

    Data augmentation: 4 pixels are padded on each side, and a 32x32 crop is randomly sampled
    from the padded image or its horizontal flip.
    This is similar to [1] and some other work that use CIFAR10.

    [1] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu. Deeply Supervised Nets.
    arXiv:1409.5185, 2014
    """
    (data_dir, args) = data

    if load_train:
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True,
                                                          transform=train_transform)
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True,
                                                         transform=test_transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'MNIST',
        'input': (1, 28, 28),
        'output': list(map(str, range(10))),
        'loader': mnist_get_datasets,
    },
    {
        'name': 'FashionMNIST',
        'input': (1, 28, 28),
        'output': ('top', 'trouser', 'pullover', 'dress', 'coat', 'sandal',
                   'shirt', 'sneaker', 'bag', 'boot'),
        'loader': fashionmnist_get_datasets,
    },
]
