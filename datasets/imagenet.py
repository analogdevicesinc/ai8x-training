#
# Copyright (c) 2018 Intel Corporation
# Portions Copyright (C) 2019-2024 Maxim Integrated Products, Inc.
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
ImageNet Dataset (using PyTorch's ImageNet and ImageFolder classes)
"""
import os

import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

import ai8x


def imagenet_get_datasets(data, load_train=True, load_test=True,
                          input_size=112, folder=False, augment_data=True):
    """
    Load the ImageNet 2012 Classification dataset.

    The original training dataset is split into training and validation sets.
    By default we use a 90:10 (45K:5K) training:validation split.

    The output of torchvision datasets are PIL Image images of range [0, 1].
    We transform them to Tensors of normalized range [-128/128, +127/128]

    Data augmentation: 4 pixels are padded on each side, and a 224x224 crop is randomly sampled
    from the padded image or its horizontal flip.
    """
    (data_dir, args) = data

    if augment_data:
        if load_train:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size, antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ai8x.normalize(args=args),
            ])

            if not folder:
                train_dataset = torchvision.datasets.ImageNet(
                    os.path.join(data_dir, 'ImageNet'),
                    split='train',
                    transform=train_transform,
                )
            else:
                train_dataset = torchvision.datasets.ImageFolder(
                    os.path.join(data_dir, 'ImageNet', 'train'),
                    transform=train_transform,
                )
        else:
            train_dataset = None

        if load_test:
            test_transform = transforms.Compose([
                transforms.Resize(int(input_size / 0.875), antialias=True),  # type: ignore
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                ai8x.normalize(args=args),
            ])

            if not folder:
                test_dataset = torchvision.datasets.ImageNet(
                    os.path.join(data_dir, 'ImageNet'),
                    split='val',
                    transform=test_transform,
                )
            else:
                test_dataset = torchvision.datasets.ImageFolder(
                    os.path.join(data_dir, 'ImageNet', 'val'),
                    transform=test_transform,
                )

            if args.truncate_testset:
                test_dataset.data = test_dataset.data[:1]  # type: ignore # .data exists
        else:
            test_dataset = None

    else:
        if load_train:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size, antialias=True),
                transforms.ToTensor(),
                ai8x.normalize(args=args),
            ])

            if not folder:
                train_dataset = torchvision.datasets.ImageNet(
                    os.path.join(data_dir, 'ImageNet'),
                    split='train',
                    transform=train_transform,
                )
            else:
                train_dataset = torchvision.datasets.ImageFolder(
                    os.path.join(data_dir, 'ImageNet', 'train'),
                    transform=train_transform,
                )
        else:
            train_dataset = None

        if load_test:
            test_transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size, antialias=True),
                transforms.ToTensor(),
                ai8x.normalize(args=args),
            ])

            if not folder:
                test_dataset = torchvision.datasets.ImageNet(
                    os.path.join(data_dir, 'ImageNet'),
                    split='val',
                    transform=test_transform,
                )
            else:
                test_dataset = torchvision.datasets.ImageFolder(
                    os.path.join(data_dir, 'ImageNet', 'val'),
                    transform=test_transform,
                )

            if args.truncate_testset:
                test_dataset.data = test_dataset.data[:1]  # type: ignore # .data exists
        else:
            test_dataset = None

    return train_dataset, test_dataset


def imagenetfolder_get_datasets(data, load_train=True, load_test=True, input_size=112):
    """
    Load the ImageNet 2012 Classification dataset using ImageFolder.
    _This function is used when the number of output classes is less than the default and
    it depends on a custom installation._
    """
    return imagenet_get_datasets(data, load_train, load_test, input_size, folder=True)


class Bayer_Dataset_Adapter(Dataset):
    """
    Implement the transforms to generate bayer filtered images from RGB images,
    and fold the input data.
    Change the target data as the input images.
    """
    def __init__(self, dataset, fold_ratio):
        self.dataset = dataset
        self.fold_ratio = fold_ratio
        self.transform = transforms.Compose([ai8x.bayer_filter(),
                                             ai8x.fold(fold_ratio=fold_ratio),
                                             ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        data = self.transform(image)
        return data, image


def imagenet_bayer_fold_2_get_dataset(data, load_train=True, load_test=True, fold_ratio=2):
    """
    Load the ImageNet 2012 Classification dataset using ImageNet.
    This function is used to modify the image dataset for debayerization network.
    Obtain raw images  from RGB images.
    """

    train_dataset, test_dataset = imagenet_get_datasets(
        data, load_train, load_test, input_size=128, augment_data=False
    )

    if load_train:
        train_dataset = Bayer_Dataset_Adapter(train_dataset, fold_ratio=fold_ratio)

    if load_test:
        test_dataset = Bayer_Dataset_Adapter(test_dataset, fold_ratio=fold_ratio)

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'ImageNet',
        'input': (3, 112, 112),
        'output': list(map(str, range(1000))),
        'loader': imagenet_get_datasets,
    },
    {
        'name': 'ImageNet50',
        'input': (3, 224, 224),
        'output': list(map(str, range(50))),
        'loader': imagenetfolder_get_datasets,
    },
    {
        'name': 'ImageNet_Bayer',
        'input': (4, 64, 64),
        'output': ('rgb'),
        'loader': imagenet_bayer_fold_2_get_dataset,
    }
]
