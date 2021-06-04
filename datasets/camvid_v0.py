###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
import os
from matplotlib.image import imread
import numpy as np

from torch.utils import data
from torchvision import transforms

import ai8x


class CamVidDataset(data.Dataset):
    def __init__(self, root_dir, d_type, download=True, transform=None, im_size=80, im_step=52):
        self.train_data_samples_per_ten_imgs = 8
        self.transform = transform
        
        img_folder = os.path.join(root_dir, 'Images')
        lbl_folder = os.path.join(root_dir, 'Labels')
        
        img_file_list = sorted([d for d in os.listdir(img_folder)])
        
        self.img_list = []
        self.lbl_list = []
        
        max_dim = 1
        
        for img_idx, img_file in enumerate(img_file_list):
            if d_type == 'train':
                if (img_idx % 10) > self.train_data_samples_per_ten_imgs:
                    continue
            else:
                if (img_idx % 10) <= self.train_data_samples_per_ten_imgs:
                    continue
            
            img = imread(os.path.join(img_folder, img_file))
            data_name = os.path.splitext(img_file)[0].rsplit('_', 1)[0]
            lbl = 255*imread(os.path.join(lbl_folder, data_name + '_Lab.png'))
            lbl = lbl.astype(np.uint8)

            x_start = y_start = 0
            while y_start < img.shape[0]:
                y_end = y_start + im_size
                if y_end >= img.shape[0]:
                    break
                while x_start < img.shape[1]:
                    x_end = x_start + im_size
                    if x_end >= img.shape[1]:
                        break
                    img_crop = img[y_start:y_end, x_start:x_end, :]
                    lbl_crop = lbl[y_start:y_end, x_start:x_end]
                    self.img_list.append(img_crop)
                    self.lbl_list.append(lbl_crop)
                    x_start = x_end
                y_start=y_end

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

        train_dataset = CamVidDataset(root_dir=os.path.join(data_dir, 'CamVid'), d_type='train', download=True, transform=train_transform)
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = CamVidDataset(root_dir=os.path.join(data_dir, 'CamVid'), d_type='test', download=True, transform=test_transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'CamVid_v0',
        'input': (3, 80, 80),
        'output': (0, 1, 2),
        'loader': camvid_get_datasets,
    },
]