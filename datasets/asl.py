###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
ASL Datasets
"""
import os
import sys
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
    if not is_dir:
        print("******************************************")
        print("Please follow instructions below:")
        print("Download the dataset in the current working directory by visiting this link"
              "https://www.kaggle.com/grassknoted/asl-alphabet")
        print("and click the \'Download (1 GB)\' button")
        print("If you do not have a Kaggle account, sign-up first.")
        print("Unzip 'asl_alphabet_test.zip' and  'asl_alphabet_test.zip.'"
              "Rename these folders to test and train respectivley"
              "and store in folder labeled 'asl_big'."
              "Delete space folder since it was not used."
              "More files can be added from train to test folders, if needed.")
        print("******************************************")
        sys.exit("Dataset not found..")
    training_data_path = os.path.join(data_dir, "asl_big")
    training_data_path = os.path.join(training_data_path, "train")
    test_data_path = os.path.join(data_dir, "asl_big")
    test_data_path = os.path.join(test_data_path, "test")
    # Loading and normalizing train dataset
    if load_train:
        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ColorJitter(
                brightness=(0.3, .8),
                contrast=(.7, 1),
                saturation=0.2,
                ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(([
                transforms.ColorJitter(),
                ]), p=0.3),
            # transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=5),
            # transforms.RandomGrayscale(p=0.2),
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
        'name': 'asl_big',
        'input': (3, 64, 64),
        'output': ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                   'k', 'l', 'm', 'n', 'nothing', 'o', 'p', 'q', 'r', 's', 't',
                   'u', 'v', 'w', 'x', 'y', 'z'),
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1),
        'loader': asl_get_datasets,
    },
]
