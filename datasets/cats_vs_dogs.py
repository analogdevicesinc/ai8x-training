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
Cats and Dogs Datasets
"""
import os
from os import makedirs
from os import listdir
import sys
from shutil import copyfile
from random import seed
from random import random
import shutil
import torchvision
from torchvision import transforms
import ai8x

def catsdogs_get_datasets(data, load_train=True, load_test=True):
    """
    Load Cats & Dogs dataset
    """
    path = './data/cats_vs_dogs/'
    is_dir = os.path.isdir(path)
    print(is_dir)

    if not is_dir:
        path = './dogs-vs-cats/'
        is_dir = os.path.isdir(path)

        if not is_dir:
            print('******************************************')
            print('Please follow instructions below:')
            print('Download the dataset in the current working directory by visiting this link' \
                '\'https://www.kaggle.com/c/dogs-vs-cats/data\'')
            print('and click the \'Download all\' button')
            print('If you do not have a Kaggle account, sign-up first.')
            print('Unzip \'dogs-vs-cats.zip\' and you will see train.zip, test1.zip and .csv ' \
                ' file. Unzip the train.zip file and re-run the script')
            print('******************************************')
            sys.exit('Dataset not found..')
        else:
            # check if train set exists
            path = './dogs-vs-cats/train'
            is_dir = os.path.isdir(path)
            if not is_dir:
                sys.exit('Unzip \'train.zip\' file from dogs-vs-cats directory')

            # create directories
            dataset_home = 'data/dogs_vs_cats/'
            subdirs = ['train/', 'test/']
            for subdir in subdirs:
                # create label subdirectories
                labeldirs = ['dogs/', 'cats/']
                for labldir in labeldirs:
                    newdir = dataset_home + subdir + labldir
                    makedirs(newdir, exist_ok=True)

            # seed random number generator
            seed(1)
            # define ratio of pictures to use for test set
            test_ratio = 0.2
            # copy training dataset images into subdirectories
            src_directory = 'dogs-vs-cats/train/'
            for file in listdir(src_directory):
                src = src_directory + '/' + file
                dst_dir = 'train/'
                if random() < test_ratio:
                    dst_dir = 'test/'
                if file.startswith('cat'):
                    dst = dataset_home + dst_dir + 'cats/'  + file
                    copyfile(src, dst)
                elif file.startswith('dog'):
                    dst = dataset_home + dst_dir + 'dogs/'  + file
                    copyfile(src, dst)
            shutil.rmtree('./dogs-vs-cats/')
            # renaming directory
            os.rename("data/dogs_vs_cats", "data/cats_vs_dogs")

    (data_dir, args) = data
    training_data_path = os.path.join(data_dir, 'cats_vs_dogs')
    training_data_path = os.path.join(training_data_path, 'train')

    test_data_path = os.path.join(data_dir, 'cats_vs_dogs')
    test_data_path = os.path.join(test_data_path, 'test')

    #Loading and normalizing train dataset
    if load_train:
        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=5),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = torchvision.datasets.ImageFolder(root=training_data_path,
                                                         transform=train_transform)
    else:
        train_dataset = None

    #Loading and normalizing test dataset
    if load_test:
        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = torchvision.datasets.ImageFolder(root=test_data_path,
                                                        transform=test_transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset

datasets = [
    {
        'name': 'cats_vs_dogs',
        'input': (3, 64, 64),
        'output': ('cat', 'dog'),
        'loader': catsdogs_get_datasets,
    },
]
