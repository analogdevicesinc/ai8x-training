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
rock, paper, scissors dataset.
"""
import errno
import os

import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds


def rock_get_datasets(data_dir):
    """
    Load "rock" dataset
    """
    print("data dir:", data_dir)

    dataset_dir = os.path.join(data_dir, 'RockPaperScissors', 'rock.npz')

    if not os.path.exists(dataset_dir):
        print("Creating dataset")
        download_dataset(dataset_dir)

    # load dataset file
    print("Load Dataset")
    a = np.load(dataset_dir)

    # read images, data range should be -128 to 127
    train_images = a['train_images'].astype(np.int32) - 128
    train_labels = a['train_labels']
    valid_images = a['valid_images'].astype(np.int32) - 128
    valid_labels = a['valid_labels']
    test_images = a['test_images'].astype(np.int32) - 128
    test_labels = a['test_labels']

    print("train_images shape:", train_images.shape)
    print("valid_images shape:", valid_images.shape)
    print("test_images shape:", test_images.shape)

    print("train_labels shape:", train_labels.shape)
    print("valid_labels shape:", valid_labels.shape)
    print("test_labels shape:", test_labels.shape)

    print("train_images min:", train_images.min())
    print("train_image max:", train_images.max())
    # print("train_labels min:", train_labels.min())
    # print("train_labels max:", train_labels.max())

    return (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels)


def download_dataset(dataset_dir):
    """
    download setset to data_dir and scale images
    """
    # scale images
    IMG_SIZE = 64

    (train_ds, valid_ds, test_ds) = tfds.load(
        'rock_paper_scissors',
        split=['train', 'test[:50%]', 'test[50%:]'],
        batch_size=-1)

    numpy_ds = tfds.as_numpy(train_ds)
    train_images, train_labels = numpy_ds["image"], numpy_ds["label"]
    numpy_ds = tfds.as_numpy(valid_ds)
    test_images, test_labels = numpy_ds["image"], numpy_ds["label"]
    numpy_ds = tfds.as_numpy(test_ds)
    valid_images, valid_labels = numpy_ds["image"], numpy_ds["label"]

    train_images = tf.image.resize(train_images, [IMG_SIZE, IMG_SIZE], preserve_aspect_ratio=True)
    valid_images = tf.image.resize(valid_images, [IMG_SIZE, IMG_SIZE], preserve_aspect_ratio=True)
    test_images = tf.image.resize(test_images, [IMG_SIZE, IMG_SIZE], preserve_aspect_ratio=True)

    # make label shape: (n,)
    train_labels = train_labels.flatten()
    valid_labels = valid_labels.flatten()
    test_labels = test_labels.flatten()

    # save
    try:
        os.makedirs(dataset_dir[:-8])
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    np.savez(
        os.path.join(dataset_dir),
        train_images=train_images,
        train_labels=train_labels,
        valid_images=valid_images,
        valid_labels=valid_labels,
        test_images=test_images,
        test_labels=test_labels)


def get_datasets(data_dir):
    """
    generic get the dataset in form of (train_images,train_labels), (valid_images, valid_labels),
    (test_images, test_labels)
    """
    return rock_get_datasets(data_dir)


def get_classnames():
    """
    name of labels
    """
    class_names = ['Rock', 'Paper', 'Scissors']
    return class_names
