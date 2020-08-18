###################################################################################################
#
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Cats and Dogs Datasets
"""
import os
import sys
from os import listdir, makedirs
from random import random
import shutil
import torchvision
from torchvision import transforms
import ai8x


def catsdogs_get_datasets(data, load_train=True, load_test=True):
    """
    Load Cats & Dogs dataset
    """
    (data_dir, args) = data
    path = data_dir
    dataset_path = os.path.join(path, "cats_vs_dogs")
    is_dir = os.path.isdir(dataset_path)
    if not is_dir:
        path = os.getcwd()
        dataset_path = os.path.join(path, "dogs-vs-cats")
        is_dir = os.path.isdir(dataset_path)

        if not is_dir:
            print("******************************************")
            print("Please follow instructions below:")
            print("Download the dataset in the current working directory by visiting this link"
                  "\'https://www.kaggle.com/c/dogs-vs-cats/data\'")
            print("and click the \'Download all\' button")
            print("If you do not have a Kaggle account, sign-up first.")
            print("Unzip \'dogs-vs-cats.zip\' and you will see train.zip, test1.zip and .csv "
                  " file. Unzip the train.zip file and re-run the script")
            print("******************************************")
            sys.exit("Dataset not found..")
        else:
            # check if train set exists
            path = os.getcwd()
            dataset_path = os.path.join(path, "dogs-vs-cats", "train")
            is_dir = os.path.isdir(dataset_path)
            if not is_dir:
                sys.exit("Unzip \'train.zip\' file from dogs-vs-cats directory")

            # create directories
            dataset_home = os.path.join(data_dir, "cats_vs_dogs")
            newdir = os.path.join(dataset_home, "train", "dogs")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "train", "cats")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "test", "dogs")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "test", "cats")
            makedirs(newdir, exist_ok=True)

            # define ratio of pictures to use for test set
            test_ratio = 0.2
            # copy training dataset images into subdirectories
            src_directory = os.path.join(path, "dogs-vs-cats", "train")
            for file in listdir(src_directory):
                src = os.path.join(src_directory, file)
                dst_dir = os.path.join(dataset_home, "train")
                if random() < test_ratio:
                    dst_dir = os.path.join(dataset_home, "test")
                if file.startswith("cat"):
                    dst = os.path.join(dst_dir, "cats", file)
                    shutil.copyfile(src, dst)
                elif file.startswith("dog"):
                    dst = os.path.join(dst_dir, "dogs", file)
                    shutil.copyfile(src, dst)
            shutil.rmtree("dogs-vs-cats")

    training_data_path = os.path.join(data_dir, "cats_vs_dogs")
    training_data_path = os.path.join(training_data_path, "train")

    test_data_path = os.path.join(data_dir, "cats_vs_dogs")
    test_data_path = os.path.join(test_data_path, "test")

    # Loading and normalizing train dataset
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

    # Loading and normalizing test dataset
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
