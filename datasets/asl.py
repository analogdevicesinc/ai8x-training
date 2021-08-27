###################################################################################################
#
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Wildlife Datasets
"""
import os
import shutil
import sys
import PIL
from os import listdir, makedirs
from random import random

import torch
import torchvision
from torchvision import transforms

import ai8x


def asl_get_datasets(data, load_train=True, load_test=True):
    """
    asl dataset
    """
    (data_dir, args) = data
    path = data_dir
    dataset_path = os.path.join(path, "asl_big")
    is_dir = os.path.isdir(dataset_path)


    training_data_path = os.path.join(data_dir, "asl_big")
    training_data_path = os.path.join(training_data_path, "train")


    test_data_path = os.path.join(data_dir, "asl_big")
    test_data_path = os.path.join(test_data_path, "test")
    print("------------")
    print(torch.rand(1))
    print(torch.rand(1))
    print(torch.rand(1))
    print(torch.rand(1))
    print("------------")
    # Loading and normalizing train dataset
    if load_train:
        train_transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            #transforms.RandomCrop((55, 55)),
            #transforms.Resize((64, 64)),
            transforms.ColorJitter(
                brightness=(0.3, .8), #*torch.randn(0, 1),
                contrast=(.7, 1), #*torch.rand(1),
               saturation=0.2, #*torch.randn(0, 1),
                ), #*torch.randn(0, 1)),
            transforms.RandomHorizontalFlip(p=0.5),
            ##transforms.RandomVerticalFlip(p=0.5),
            ##transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
            transforms.RandomApply(([
                transforms.ColorJitter(),
                ]), p=0.3),
            #transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=5),
            #transforms.RandomGrayscale(p=0.2),
            #transforms.RandomCrop((64, 64)),
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
            #transforms.Resize((32, 32)),
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
        'name': 'asl_big',
        'input': (3, 64, 64), 
        #'input': (3, 32, 32),
        'output': ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                   'k', 'l', 'm', 'n', 'nothing', 'o', 'p', 'q', 'r', 's', 't',
                   'u', 'v', 'w', 'x', 'y', 'z'),
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1),
       #'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        #'output': ('cow', 'Horse', 'human'),
        'loader': asl_get_datasets,
    },
]


