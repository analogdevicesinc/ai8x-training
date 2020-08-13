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
fashionmnist dataset.
"""
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np


def fashionmnist_get_datasets(data_dir):
    """
    Load "mnist" dataset
    """
    print("data dir:", data_dir)
    fashionmnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images,
                                   test_labels) = fashionmnist.load_data()

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
    return fashionmnist_get_datasets(data_dir)


def get_classnames():
    """
    name of labels
    """
    class_names = [
        'top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'boot'
    ]
    return class_names
