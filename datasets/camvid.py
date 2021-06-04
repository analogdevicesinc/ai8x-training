###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Classes and functions used to create CamVid dataset.
"""
import os
import csv
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms

import ai8x

class CamVidDataset(data.Dataset):
    """
    Possible class selections:
    Animal, Archway, Bicyclist, Bridge, Building, Car, CartLuggagePram, Child, Column_Pole, Fence,
    LaneMkgsDriv, LaneMkgsNonDriv, Misc_Text, MotorcycleScooter, OtherMoving, ParkingBlock,
    Pedestrian, Road, RoadShoulder, Sidewalk, SignSymbol, Sky, SUVPickupTruck, TrafficCone,
    TrafficLight, Train, Tree, Truck_Bus, Tunnel, VegetationMisc, Void, Wall
    """
    
    def __init__(self, root_dir, d_type, classes=None, download=True, transform=None, im_scale=1, im_size=[80, 80], im_overlap=[20, 20]):
        self.transform = transform
        
        img_dims = [720//im_scale, 960//im_scale]
        img_folder = os.path.join(root_dir, d_type)
        lbl_folder = os.path.join(root_dir, d_type + '_labels')
        class_dict_file = os.path.join(root_dir, 'class_dict.csv')
        
        self.label_mask_dict = {}
        self.__create_mask_dict(class_dict_file, classes, img_dims)
        
        img_file_list = sorted([d for d in os.listdir(img_folder)])
        
        self.img_list = []
        self.lbl_list = []
        
        max_dim = 1
        
        for img_idx, img_file in enumerate(img_file_list):            
            img = np.asarray(Image.open(os.path.join(img_folder, img_file)))[::im_scale, ::im_scale, :]
            data_name = os.path.splitext(img_file)[0]
            lbl_rgb = np.asarray(Image.open(os.path.join(lbl_folder, data_name + '_L.png')))[::im_scale, ::im_scale, :]
            lbl = np.zeros((lbl_rgb.shape[0], lbl_rgb.shape[1]), dtype=np.uint8)
            
            for label_idx, (label, mask) in enumerate(self.label_mask_dict.items()):
                res = (lbl_rgb == mask)
                res = (label_idx+1) * res.all(axis=2)
                lbl += res.astype(np.uint8)

            x_start = y_start = 0
            while y_start < img.shape[0]:
                y_end = y_start + im_size[0]
                if y_end >= img.shape[0]:
                    break
                while x_start < img.shape[1]:
                    x_end = x_start + im_size[1]
                    if x_end >= img.shape[1]:
                        break
                    img_crop = img[y_start:y_end, x_start:x_end, :]
                    lbl_crop = lbl[y_start:y_end, x_start:x_end]
                    self.img_list.append(img_crop)
                    self.lbl_list.append(lbl_crop)
                    x_start = x_end - im_overlap[1]
                y_start = y_end - im_overlap[0]

    def __create_mask_dict(self, class_dict_file, classes, img_dims):
        with open(class_dict_file, newline='') as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                if row[0] == 'name':
                    continue
                if classes:
                    if row[0] not in classes:
                        continue

                label = row[0]
                label_mask = np.zeros((img_dims[0], img_dims[1], 3), dtype=np.uint8)
                label_mask[:, :, 0] = np.uint8(row[1])
                label_mask[:, :, 1] = np.uint8(row[2])
                label_mask[:, :, 2] = np.uint8(row[3])

                self.label_mask_dict[label] = label_mask

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.transform is not None:
            img = self.transform(self.img_list[idx])
        return img, self.lbl_list[idx].astype(np.long)
    

def camvid_get_datasets(data, load_train=True, load_test=True):
    (data_dir, args) = data

    if load_train:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = CamVidDataset(root_dir=os.path.join(data_dir, 'CamVid'), d_type='train', classes=None, download=True, transform=train_transform)
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = CamVidDataset(root_dir=os.path.join(data_dir, 'CamVid'), d_type='test', classes=None, download=True, transform=test_transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'CamVidAll',
        'input': (3, 80, 80),
        'output': ('Animal', 'Archway', 'Bicyclist', 'Bridge', 'Building', 'Car',
                   'CartLuggagePram', 'Child', 'Column_Pole', 'Fence', 'LaneMkgsDriv',
                   'LaneMkgsNonDriv', 'Misc_Text', 'MotorcycleScooter', 'OtherMoving',
                   'ParkingBlock', 'Pedestrian', 'Road', 'RoadShoulder', 'Sidewalk', 'SignSymbol',
                   'Sky', 'SUVPickupTruck', 'TrafficCone', 'TrafficLight', 'Train', 'Tree',
                   'Truck_Bus', 'Tunnel', 'VegetationMisc', 'Void', 'Wall'),
        'loader': camvid_get_datasets,
    },
]