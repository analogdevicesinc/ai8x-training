###################################################################################################
#
# Copyright (C) 2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Classes and functions used to create CamVid dataset.
"""
import copy
import csv
import os
import sys

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

import ai8x


class CamVidDataset(Dataset):
    """
    Possible class selections:
    Animal, Archway, Bicyclist, Bridge, Building, Car, CartLuggagePram, Child, Column_Pole, Fence,
    LaneMkgsDriv, LaneMkgsNonDriv, Misc_Text, MotorcycleScooter, OtherMoving, ParkingBlock,
    Pedestrian, Road, RoadShoulder, Sidewalk, SignSymbol, Sky, SUVPickupTruck, TrafficCone,
    TrafficLight, Train, Tree, Truck_Bus, Tunnel, VegetationMisc, Void, Wall
    """

    class_dict = {'None': 0, 'Animal': 1, 'Archway': 2, 'Bicyclist': 3, 'Bridge': 4, 'Building': 5,
                  'Car': 6, 'CartLuggagePram': 7, 'Child': 8, 'Column_Pole': 9, 'Fence': 10,
                  'LaneMkgsDriv': 11, 'LaneMkgsNonDriv': 12, 'Misc_Text': 13,
                  'MotorcycleScooter': 14, 'OtherMoving': 15, 'ParkingBlock': 16, 'Pedestrian': 17,
                  'Road': 18, 'RoadShoulder': 19, 'Sidewalk': 20, 'SignSymbol': 21, 'Sky': 22,
                  'SUVPickupTruck': 23, 'TrafficCone': 24, 'TrafficLight': 25, 'Train': 26,
                  'Tree': 27, 'Truck_Bus': 28, 'Tunnel': 29, 'VegetationMisc': 30, 'Void': 31,
                  'Wall': 32}

    def __init__(self, root_dir, d_type, classes=None, download=True, transform=None, im_scale=1,
                 im_size=(80, 80), im_overlap=(20, 20)):
        self.transform = transform
        self.classes = classes

        img_dims = [720//im_scale, 960//im_scale]
        img_folder = os.path.join(root_dir, d_type)
        lbl_folder = os.path.join(root_dir, d_type + '_labels')
        self.class_dict_file = os.path.join(root_dir, 'class_dict.csv')
        self.img_list = []
        self.lbl_list = []

        if download:
            if not self.__download():
                sys.exit()

        self.label_mask_dict = {}
        self.__create_mask_dict(img_dims)

        img_file_list = sorted(os.listdir(img_folder))

        for img_file in img_file_list:
            img = np.asarray(Image.open(os.path.join(img_folder, img_file)))
            if im_scale != 1:
                img = img[::im_scale, ::im_scale, :]
            img = CamVidDataset.normalize(img.astype(np.float32))
            data_name = os.path.splitext(img_file)[0]
            lbl_rgb = np.asarray(Image.open(os.path.join(lbl_folder, data_name + '_L.png')))
            if im_scale != 1:
                lbl_rgb = lbl_rgb[::im_scale, ::im_scale, :]
            lbl = np.zeros((lbl_rgb.shape[0], lbl_rgb.shape[1]), dtype=np.uint8)

            for label_idx, (_, mask) in enumerate(self.label_mask_dict.items()):
                res = lbl_rgb == mask
                res = (label_idx + 1) * res.all(axis=2)
                lbl += res.astype(np.uint8)

            y_start = 0
            while y_start < img.shape[0]:
                x_start = 0
                y_end = y_start + im_size[0]
                if y_end > img.shape[0]:
                    break
                while x_start < img.shape[1]:
                    x_end = x_start + im_size[1]
                    if x_end > img.shape[1]:
                        break

                    img_crop = copy.deepcopy(img[y_start:y_end, x_start:x_end, :])
                    lbl_crop = copy.deepcopy(lbl[y_start:y_end, x_start:x_end])
                    self.img_list.append(img_crop)
                    self.lbl_list.append(lbl_crop)

                    x_start = x_end - im_overlap[1]
                y_start = y_end - im_overlap[0]

        if self.classes:
            self.__filter_classes()

    def __download(self):
        if self.__check_exists():
            return True

        print('Download the archive file from https://www.kaggle.com/carlolepelaars/camvid/'
              'download and extract to path data/CamVid. The download process may require '
              'additional authentication.')
        return False

    def __check_exists(self):
        return os.path.exists(self.class_dict_file)

    def __create_mask_dict(self, img_dims):
        with open(self.class_dict_file, newline='', encoding='utf-8') as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                if row[0] == 'name':
                    continue

                label = row[0]
                label_mask = np.zeros((img_dims[0], img_dims[1], 3), dtype=np.uint8)
                label_mask[:, :, 0] = np.uint8(row[1])
                label_mask[:, :, 1] = np.uint8(row[2])
                label_mask[:, :, 2] = np.uint8(row[3])

                self.label_mask_dict[label] = label_mask

    def __filter_classes(self):
        for e in self.lbl_list:
            initial_new_class_label = len(self.class_dict) + 5
            new_class_label = initial_new_class_label
            for l_class in self.classes:
                if l_class not in self.class_dict:
                    print(f'Class is not in the data: {l_class}')
                    return

                e[(e == self.class_dict[l_class])] = new_class_label
                new_class_label += 1

            e[(e < initial_new_class_label)] = new_class_label
            e -= initial_new_class_label

    @staticmethod
    def normalize(data):
        """Normalizes data to the range [0, 1)"""
        return data / 256.

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.transform is not None:
            img = self.transform(self.img_list[idx])
        return img, self.lbl_list[idx].astype(np.int64)


def camvid_get_datasets_s80(data, load_train=True, load_test=True, num_classes=33):
    """
    Load the CamVid dataset in 3x80x80 size.

    The dataset originally includes 33 keywords. A dataset is formed with 4 or 34 classes which
    includes 3, or 33 of the original keywords and the rest of the dataset is used to form the
    last class, i.e class of the others.

    The dataset is split into training+validation and test sets. 90:10 training+validation:test
    split is used by default.
    """
    (data_dir, args) = data

    if num_classes == 3:
        classes = ['Building', 'Sky', 'Tree']
    elif num_classes == 33:
        classes = None
    else:
        raise ValueError(f'Unsupported num_classes {num_classes}')

    if load_train:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = CamVidDataset(root_dir=os.path.join(data_dir, 'CamVid'), d_type='train',
                                      im_size=[80, 80], im_overlap=[20, 20], classes=classes,
                                      download=True, transform=train_transform)
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = CamVidDataset(root_dir=os.path.join(data_dir, 'CamVid'), d_type='test',
                                     im_size=[80, 80], im_overlap=[20, 20], classes=classes,
                                     download=True, transform=test_transform)

        if args.truncate_testset:
            test_dataset.img_list = test_dataset.img_list[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


def camvid_get_datasets_s352(data, load_train=True, load_test=True, num_classes=33):
    """
    Load the CamVid dataset in 48x88x88 format which are composed of 3x352x352 images folded
    with a fold_ratio of 4.

    The dataset originally includes 33 keywords. A dataset is formed with 4 or 34 classes which
    includes 3, or 33 of the original keywords and the rest of the dataset is used to form the
    last class, i.e class of the others.

    The dataset is split into training+validation and test sets. 90:10 training+validation:test
    split is used by default.
    """
    (data_dir, args) = data

    if num_classes == 3:
        classes = ['Building', 'Sky', 'Tree']
    elif num_classes == 33:
        classes = None
    else:
        raise ValueError(f'Unsupported num_classes {num_classes}')

    transform = transforms.Compose([transforms.ToTensor(),
                                    ai8x.normalize(args=args),
                                    ai8x.fold(fold_ratio=4)])

    if load_train:
        train_dataset = CamVidDataset(root_dir=os.path.join(data_dir, 'CamVid'), d_type='train',
                                      im_size=[352, 352], im_overlap=[168, 150], classes=classes,
                                      download=True, transform=transform)
    else:
        train_dataset = None

    if load_test:
        test_dataset = CamVidDataset(root_dir=os.path.join(data_dir, 'CamVid'), d_type='test',
                                     im_size=[352, 352], im_overlap=[0, 54], classes=classes,
                                     download=True, transform=transform)

        if args.truncate_testset:
            test_dataset.img_list = test_dataset.img_list[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


def camvid_get_datasets_s80_c3(data, load_train=True, load_test=True):
    """
    Load the Camvid dataset for 3 classes in 3x80x80 images.
    """
    return camvid_get_datasets_s80(data, load_train, load_test, num_classes=3)


def camvid_get_datasets_s80_c33(data, load_train=True, load_test=True):
    """
    Load the Camvid dataset for 33 classes in 3x80x80 images.
    """
    return camvid_get_datasets_s80(data, load_train, load_test, num_classes=33)


def camvid_get_datasets_s352_c3(data, load_train=True, load_test=True):
    """
    Load the Camvid dataset for 3 classes in 48x88x88 images.
    """
    return camvid_get_datasets_s352(data, load_train, load_test, num_classes=3)


def camvid_get_datasets_s352_c33(data, load_train=True, load_test=True):
    """
    Load the Camvid dataset for 33 classes in 48x88x88 images.
    """
    return camvid_get_datasets_s352(data, load_train, load_test, num_classes=33)


datasets = [
    {
        'name': 'CamVid_s80_c33',
        'input': (3, 80, 80),
        'output': ('None', 'Animal', 'Archway', 'Bicyclist', 'Bridge', 'Building', 'Car',
                   'CartLuggagePram', 'Child', 'Column_Pole', 'Fence', 'LaneMkgsDriv',
                   'LaneMkgsNonDriv', 'Misc_Text', 'MotorcycleScooter', 'OtherMoving',
                   'ParkingBlock', 'Pedestrian', 'Road', 'RoadShoulder', 'Sidewalk', 'SignSymbol',
                   'Sky', 'SUVPickupTruck', 'TrafficCone', 'TrafficLight', 'Train', 'Tree',
                   'Truck_Bus', 'Tunnel', 'VegetationMisc', 'Void', 'Wall'),
        'loader': camvid_get_datasets_s80_c33,
    },
    {
        'name': 'CamVid_s80_c3',  # 3 classes
        'input': (3, 80, 80),
        'output': (0, 1, 2, 3),
        'weight': (1, 1, 1, 0.14),
        'loader': camvid_get_datasets_s80_c3,
    },
    {
        'name': 'CamVid_s352_c33',
        'input': (48, 88, 88),
        'output': ('None', 'Animal', 'Archway', 'Bicyclist', 'Bridge', 'Building', 'Car',
                   'CartLuggagePram', 'Child', 'Column_Pole', 'Fence', 'LaneMkgsDriv',
                   'LaneMkgsNonDriv', 'Misc_Text', 'MotorcycleScooter', 'OtherMoving',
                   'ParkingBlock', 'Pedestrian', 'Road', 'RoadShoulder', 'Sidewalk', 'SignSymbol',
                   'Sky', 'SUVPickupTruck', 'TrafficCone', 'TrafficLight', 'Train', 'Tree',
                   'Truck_Bus', 'Tunnel', 'VegetationMisc', 'Void', 'Wall'),
        'loader': camvid_get_datasets_s352_c33,
        'fold_ratio': 4,
    },
    {
        'name': 'CamVid_s352_c3',  # 3 classes
        'input': (48, 88, 88),
        'output': (0, 1, 2, 3),
        'weight': (1, 1, 1, 1),
        'loader': camvid_get_datasets_s352_c3,
        'fold_ratio': 4,
    }
]
