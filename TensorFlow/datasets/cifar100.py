###################################################################################################
#
# Copyright (C) 2018-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
#
# Portions Copyright (c) 2018 Intel Corporation
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
cifar100 dataset.
"""
from tensorflow.keras import datasets
from sklearn.model_selection import train_test_split
import numpy as np


def cifar100_get_datasets(data_dir):
    """
    Load "cifar100" dataset
    """
    print("data dir:", data_dir)
    (train_images,
     train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

    # split to train, valid and test
    train_images, valid_images, train_labels, valid_labels = train_test_split(
        train_images, train_labels, test_size=0.1, random_state=42)

    # read images, data range should be -128 to 127
    train_images = train_images.astype(np.int32) - 128
    valid_images = valid_images.astype(np.int32) - 128
    test_images = test_images.astype(np.int32) - 128

    # make label shape: (n,)
    train_labels = train_labels.flatten()
    valid_labels = valid_labels.flatten()
    test_labels = test_labels.flatten()

    return (train_images, train_labels), (valid_images,
                                          valid_labels), (test_images,
                                                          test_labels)


def get_datasets(data_dir):
    """
    generic get the dataset in form of (train_images,train_labels), (valid_images, valid_labels),
    (test_images, test_labels)
    """
    return cifar100_get_datasets(data_dir)


def get_classnames():
    """
    name of labels
    """
    class_names = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
        'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
        'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab',
        'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
        'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
        'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom',
        'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
        'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
        'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
        'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
        'willow_tree', 'wolf', 'woman', 'worm'
    ]
    return class_names
