###################################################################################################
#
# Copyright (C) 2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Cats and Dogs Datasets
"""
import errno
import os
import shutil
import sys

import torch
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms

import albumentations as album
from albumentations.pytorch import ToTensorV2

import cv2

from PIL import Image

import ai8x

torch.manual_seed(0)


def augment_affine_jitter_blur(orig_img):
    """
    Augment with multiple transformations
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.2),
        transforms.CenterCrop((180, 180)),
        transforms.ColorJitter(brightness=.7),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
        transforms.RandomHorizontalFlip(),
        ])
    return train_transform(orig_img)


def augment_blur(orig_img):
    """
    Augment with center crop and bluring
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((220, 220)),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))
        ])
    return train_transform(orig_img)


def catsdogs_get_datasets(data, load_train=True, load_test=True, aug=2):
    """
    Load Cats & Dogs dataset
    """
    (data_dir, args) = data
    path = data_dir
    dataset_path = os.path.join(path, "cats_vs_dogs")
    is_dir = os.path.isdir(dataset_path)
    if not is_dir:
        print("******************************************")
        print("Please follow the instructions below:")
        print("Download the dataset to the \'data\' folder by visiting this link: "
              "\'https://www.kaggle.com/datasets/salader/dogs-vs-cats\'")
        print("If you do not have a Kaggle account, sign up first.")
        print("Unzip the downloaded file and find \'test\' and \'train\' folders "
              "and copy them into \'data/cats_vs_dogs\'. ")
        print("Make sure that images are in the following directory structure:")
        print("  \'data/cats_vs_dogs/train/cats\'")
        print("  \'data/cats_vs_dogs/train/dogs\'")
        print("  \'data/cats_vs_dogs/test/cats\'")
        print("  \'data/cats_vs_dogs/test/dogs\'")
        print("Re-run the script. The script will create an \'augmented\' folder ")
        print("with all the original and augmented images. Remove this folder if you want "
              "to change the augmentation and to recreate the dataset.")
        print("******************************************")
        sys.exit("Dataset not found!")
    else:
        processed_dataset_path = os.path.join(dataset_path, "augmented")

        if os.path.isdir(processed_dataset_path):
            print("augmented folder exits. Remove if you want to regenerate")

        train_path = os.path.join(dataset_path, "train")
        test_path = os.path.join(dataset_path, "test")
        processed_train_path = os.path.join(processed_dataset_path, "train")
        processed_test_path = os.path.join(processed_dataset_path, "test")
        if not os.path.isdir(processed_dataset_path):
            os.makedirs(processed_dataset_path, exist_ok=True)
            os.makedirs(processed_test_path, exist_ok=True)
            os.makedirs(processed_train_path, exist_ok=True)

            # create label folders
            for d in os.listdir(test_path):
                mk = os.path.join(processed_test_path, d)
                try:
                    os.mkdir(mk)
                except OSError as e:
                    if e.errno == errno.EEXIST:
                        print(f'{mk} already exists!')
                    else:
                        raise
            for d in os.listdir(train_path):
                mk = os.path.join(processed_train_path, d)
                try:
                    os.mkdir(mk)
                except OSError as e:
                    if e.errno == errno.EEXIST:
                        print(f'{mk} already exists!')
                    else:
                        raise

            # copy test folder files
            test_cnt = 0
            for (dirpath, _, filenames) in os.walk(test_path):
                print(f'copying {dirpath} -> {processed_test_path}')
                for filename in filenames:
                    if filename.endswith('.jpg'):
                        relsourcepath = os.path.relpath(dirpath, test_path)
                        destpath = os.path.join(processed_test_path, relsourcepath)

                        destfile = os.path.join(destpath, filename)
                        shutil.copyfile(os.path.join(dirpath, filename), destfile)
                        test_cnt += 1

            # copy and augment train folder files
            train_cnt = 0
            for (dirpath, _, filenames) in os.walk(train_path):
                print(f'copying and augmenting {dirpath} -> {processed_train_path}')
                for filename in filenames:
                    if filename.endswith('.jpg'):
                        relsourcepath = os.path.relpath(dirpath, train_path)
                        destpath = os.path.join(processed_train_path, relsourcepath)
                        srcfile = os.path.join(dirpath, filename)
                        destfile = os.path.join(destpath, filename)

                        # original file
                        shutil.copyfile(srcfile, destfile)
                        train_cnt += 1

                        orig_img = Image.open(srcfile)

                        # crop center & blur only
                        aug_img = augment_blur(orig_img)
                        augfile = destfile[:-4] + '_ab' + str(0) + '.jpg'
                        aug_img.save(augfile)
                        train_cnt += 1

                        # random jitter, affine, brightness & blur
                        for i in range(aug):
                            aug_img = augment_affine_jitter_blur(orig_img)
                            augfile = destfile[:-4] + '_aj' + str(i) + '.jpg'
                            aug_img.save(augfile)
                            train_cnt += 1
            print(f'Augmented dataset: {test_cnt} test, {train_cnt} train samples')

    # Loading and normalizing train dataset
    if load_train:
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = torchvision.datasets.ImageFolder(root=processed_train_path,
                                                         transform=train_transform)
    else:
        train_dataset = None

    # Loading and normalizing test dataset
    if load_test:
        test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = torchvision.datasets.ImageFolder(root=processed_test_path,
                                                        transform=test_transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


class CatsvsDogs(Dataset):
    labels = ['cat', 'dog']
    label_to_id_map = {k: v for v, k in enumerate(labels)}
    label_to_folder_map = {'cat': 'cats', 'dog': 'dogs'}

    def __init__(self, root_dir, d_type, transform=None, resize_size=(128, 128), augment_data=False):
        self.root_dir = root_dir
        self.data_dir =  os.path.join(root_dir, 'cats_vs_dogs', d_type)
        
        if not self.__check_catsvsdogs_data_exist():
            self.__print_download_manual()
            sys.exit("Dataset not found!")
        
        self.__get_image_paths()
        
        self.album_transform =  None
        if d_type == 'train' and augment_data:
            self.album_transform =  train_transform = album.Compose([
                album.GaussNoise (var_limit=(1.0, 20.0), p=0.25),
                album.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                album.ColorJitter(p=0.5),
                album.SmallestMaxSize(max_size=int(1.2*min(resize_size))),
                album.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                album.RandomCrop(height=resize_size[0], width=resize_size[1]),
                album.HorizontalFlip(p=0.5),
                album.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))]) # converts range from [0,255] to [0, 1)
        if not augment_data or d_type=='test':
            self.album_transform =  train_transform = album.Compose([
                album.SmallestMaxSize(max_size=int(1.2*min(resize_size))),
                album.CenterCrop(height=resize_size[0], width=resize_size[1]),
                album.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))]) # converts range from [0,255] to [0, 1)

        
        self.transform = transform
        
    def __check_catsvsdogs_data_exist(self):
        return os.path.isdir(self.data_dir)
    
    def __print_download_manual(self):    
        print("******************************************")
        print("Please follow the instructions below:")
        print("Download the dataset to the \'data\' folder by visiting this link: "
              "\'https://www.kaggle.com/datasets/salader/dogs-vs-cats\'")
        print("If you do not have a Kaggle account, sign up first.")
        print("Unzip the downloaded file and find \'test\' and \'train\' folders "
              "and copy them into \'data/cats_vs_dogs\'. ")
        print("Make sure that images are in the following directory structure:")
        print("  \'data/cats_vs_dogs/train/cats\'")
        print("  \'data/cats_vs_dogs/train/dogs\'")
        print("  \'data/cats_vs_dogs/test/cats\'")
        print("  \'data/cats_vs_dogs/test/dogs\'")
        print("Re-run the script. The script will create an \'augmented\' folder ")
        print("with all the original and augmented images. Remove this folder if you want "
              "to change the augmentation and to recreate the dataset.")
        print("******************************************")

    def __get_image_paths(self):
        self.data_list = []
        
        for label in self.labels:
            image_dir = os.path.join(self.data_dir, self.label_to_folder_map[label])
            for file_name in sorted(os.listdir(image_dir)):
                file_path = os.path.join(image_dir, file_name)
                if os.path.isfile(file_path):
                    self.data_list.append((file_path, self.label_to_id_map[label]))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label = torch.tensor(self.data_list[index][1], dtype=torch.int64)
        
        image_path = self.data_list[index][0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.album_transform:
            image = self.album_transform(image=image)["image"]

        #print(image.shape, image.min(), image.max(), image.dtype)
        
        if self.transform:
            image = self.transform(image)
        
        #print(image.shape, image.min(), image.max(), image.dtype)
        
        return image, label


def get_catsvsdogs_dataset(data, load_train, load_test):
    
    (data_dir, args) = data

    transform = transforms.Compose([
        transforms.ToTensor(),
        ai8x.normalize(args=args),
    ])
    
    if load_train:
        train_dataset = CatsvsDogs(root_dir=data_dir, d_type='train',
                                   transform=transform, augment_data=True)
    else:
        train_dataset = None
    
    if load_test:
        test_dataset = CatsvsDogs(root_dir=data_dir, d_type='test', transform=transform)
    else:
        test_dataset = None
    
    return train_dataset, test_dataset


datasets = [
    {
        'name': 'cats_vs_dogs',
        'input': (3, 128, 128),
        'output': ('cat', 'dog'),
        'loader': catsdogs_get_datasets,
    },
    {
        'name': 'cats_vs_dogs_v2',
        'input': (3, 128, 128),
        'output': ('cat', 'dog'),
        'loader': get_catsvsdogs_dataset,
    },
]
