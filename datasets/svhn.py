###################################################################################################
#
# Copyright (C) 2022-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Classes and functions used to create the Street View House Numbers (SVHN) Dataset.
(http://ufldl.stanford.edu/housenumbers/)
Format: 1 is used: Format with Bounding Boxes
"""
import ast
import errno
import os
import pickle
import random
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import h5py
import pandas as pd
from PIL import Image

import ai8x


class SVHN(Dataset):
    """
    Street View House Numbers (SVHN) Dataset. (http://ufldl.stanford.edu/housenumbers/)
    Format: 1 is used: Format with Bounding Boxes

    Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng Reading Digits
    in Natural Images with Unsupervised Feature Learning NIPS Workshop on Deep Learning and
    Unsupervised Feature Learning 2011.
    """

    expansion_ratio = 0.3

    def __init__(self, root_dir, d_type, transform=None, resize_size=(96, 96), fold_ratio=2,
                 simplified=False):

        if d_type not in ('test', 'train'):
            raise ValueError("d_type can only be set to 'test' or 'train'")

        if resize_size[0] != resize_size[1]:
            raise ValueError('resize_size should be square')

        self.root_dir = root_dir
        self.d_type = d_type
        self.transform = transform
        self.resize_size = resize_size
        self.fold_ratio = fold_ratio
        self.simplified = simplified

        self.img_list = []
        self.boxes_list = []
        self.lbls_list = []

        self.processed_folder = os.path.join(root_dir, self.__class__.__name__, 'processed')
        self.__makedir_exist_ok(self.processed_folder)

        res_string = str(self.resize_size[0]) + 'x' + str(self.resize_size[1])
        simplified_string = "_simplified" if self.simplified else ""

        train_pkl_file_path = os.path.join(self.processed_folder, 'train_' + res_string +
                                           '_fold_' + str(self.fold_ratio) + simplified_string +
                                           '.pkl')
        test_pkl_file_path = os.path.join(self.processed_folder, 'test_' + res_string + '_fold_' +
                                          str(self.fold_ratio) + simplified_string + '.pkl')

        if self.d_type == 'train':
            self.pkl_file = train_pkl_file_path
            self.info_df_csv_file = os.path.join(self.processed_folder, 'train_info.csv')
        elif self.d_type == 'test':
            self.pkl_file = test_pkl_file_path
            self.info_df_csv_file = os.path.join(self.processed_folder, 'test_info.csv')
        else:
            print(f'Unknown data type: {self.d_type}')
            return

        self.__create_info_df_csv()
        self.__create_pkl_file()
        self.is_truncated = False

    def __create_info_df_csv(self):

        if os.path.exists(self.info_df_csv_file):
            self.info_df = pd.read_csv(self.info_df_csv_file)

            for column in self.info_df.columns:
                if column in ['label', 'x0', 'x1', 'y0', 'y1']:
                    self.info_df[column] = \
                        self.info_df[column].apply(ast.literal_eval)
        else:

            mat_file_path = os.path.join(self.root_dir, self.__class__.__name__, self.d_type,
                                         'digitStruct.mat')

            if not os.path.exists(mat_file_path):
                print('\nDownload the archive file from: '
                      'http://ufldl.stanford.edu/housenumbers/[train or test].tar.gz\n'
                      'Review the terms and conditions on '
                      'http://ufldl.stanford.edu/housenumbers/ and then download...\n'
                      'Extract the downloaded archive to path /data/SVHN\n'
                      'E.g. The training image files and digitStruct.mat file containing all '
                      'annotations will reside under folder: /data/SVHN/training\n')
                sys.exit()

            digit_struct_data = SVHN.read_digit_mat(os.path.join(mat_file_path))
            self.info_df = digit_struct_data.groupby('img_name', as_index=False,
                                                     sort=False).agg(list)

            # Eliminate some entries as some entries (very few but has to be eliminated) have -1
            neg_indexes = self.info_df[self.info_df['x0'].apply(
                lambda list: any(n < 0 for n in list))].index

            print(f"neg_indexes: {neg_indexes}")

            # Delete these row indexes from dataFrame
            self.info_df.drop(neg_indexes, inplace=True)

            self.info_df['num_of_boxes'] = self.info_df['label'].apply(len)

            self.info_df['img_width'], self.info_df['img_height'] = \
                zip(*self.info_df['img_name'].apply(lambda name: SVHN.get_image_size(
                    os.path.join(self.root_dir, self.__class__.__name__, self.d_type, name))))

            self.info_df['bb_x0'] = self.info_df['x0'].apply(min).apply(int)
            self.info_df['bb_y0'] = self.info_df['y0'].apply(min).apply(int)
            self.info_df['bb_x1'] = self.info_df['x1'].apply(max).apply(int)
            self.info_df['bb_y1'] = self.info_df['y1'].apply(max).apply(int)

            # Save info dataframe into csv:
            self.info_df.to_csv(self.info_df_csv_file, index=False)

    def __create_pkl_file(self):

        if os.path.exists(self.pkl_file):

            (self.img_list, self.boxes_list, self.lbls_list) = \
                    pickle.load(open(self.pkl_file, 'rb'))
            return
        self.__gen_datasets()

    def __gen_datasets(self):
        print('\nGenerating dataset pickle file from the raw image files...\n')

        total_num_of_processed_files = 0

        for _, row in self.info_df.iterrows():

            img_width = row['img_width']
            img_height = row['img_height']

            rectangle_bb_width = row['bb_x1'] - row['bb_x0']
            rectangle_bb_height = row['bb_y1'] - row['bb_y0']

            if rectangle_bb_width > rectangle_bb_height:
                is_rectangle_wide = True
                square_bb_width = rectangle_bb_width
                slide_range = rectangle_bb_width - rectangle_bb_height
            else:
                is_rectangle_wide = False
                square_bb_width = rectangle_bb_height
                slide_range = rectangle_bb_height - rectangle_bb_width

            square_bb_x0_selected = row['bb_x0']
            square_bb_y0_selected = row['bb_y0']
            square_bb_x1_selected = row['bb_x1']
            square_bb_y1_selected = row['bb_y1']

            if is_rectangle_wide:
                square_bb_x0_min = row['bb_x0']

                # Only y will change
                square_bb_x0_selected = row['bb_x0']
                square_bb_x1_selected = row['bb_x1']

                square_bb_y0_min = max(0, row['bb_y0'] - slide_range)
                square_bb_y0_max = row['bb_y0']

                square_bb_y0_selected = random.randint(square_bb_y0_min, square_bb_y0_max)
                square_bb_y1_selected = square_bb_y0_selected + square_bb_width

                if square_bb_y1_selected > img_height:
                    print('y', square_bb_y1_selected, img_height)
                    print(f"{total_num_of_processed_files + 1} th image can NOT be used: smallest \
                            square including existing bounding boxes exceeds image height")
                    continue

            else:
                square_bb_y0_min = row['bb_y0']

                # Only x will change
                square_bb_y0_selected = row['bb_y0']
                square_bb_y1_selected = row['bb_y1']

                square_bb_x0_min = max(0, row['bb_x0'] - slide_range)
                square_bb_x0_max = row['bb_x0']

                square_bb_x0_selected = random.randint(square_bb_x0_min, square_bb_x0_max)
                square_bb_x1_selected = square_bb_x0_selected + square_bb_width

                if square_bb_x1_selected > img_width:
                    print('x', square_bb_x1_selected, img_width)
                    print(f"{total_num_of_processed_files + 1} th image can NOT be used: smallest \
                            square including existing bounding boxes exceeds image width")
                    continue

            # Expand square box with exp ratio in both directions, if image size permits:
            increase = round(square_bb_width * SVHN.expansion_ratio / 2.)

            expanded_square_bb_x0 = square_bb_x0_selected
            expanded_square_bb_x1 = square_bb_x1_selected
            expanded_square_bb_y0 = square_bb_y0_selected
            expanded_square_bb_y1 = square_bb_y1_selected

            # Apply the increase in all both  directions if you can, else decrease increase amount
            while increase > 1:
                if (
                    square_bb_x0_selected - increase < 0 or
                    square_bb_x1_selected + increase > img_width or
                    square_bb_y0_selected - increase < 0 or
                    square_bb_y1_selected + increase > img_height
                   ):

                    increase = increase // 2
                else:
                    expanded_square_bb_x0 = square_bb_x0_selected - increase
                    expanded_square_bb_x1 = square_bb_x1_selected + increase
                    expanded_square_bb_y0 = square_bb_y0_selected - increase
                    expanded_square_bb_y1 = square_bb_y1_selected + increase
                    break

            # Read image
            image = Image.open(os.path.join(self.root_dir, self.__class__.__name__, self.d_type,
                                            row['img_name']))

            # Crop expanded square first:
            img_crp = image.crop((expanded_square_bb_x0, expanded_square_bb_y0,
                                  expanded_square_bb_x1, expanded_square_bb_y1))

            # Resize cropped expanded square:
            img_crp_resized = img_crp.resize(self.resize_size)
            img_crp_resized = np.asarray(img_crp_resized).astype(np.uint8)

            # Fold cropped expanded square (96 x 96 x 3 folded into 48 x 48 x 12) if required:
            img_crp_resized_folded = self.fold_image(img_crp_resized, self.fold_ratio)

            self.img_list.append(img_crp_resized_folded)

            scaling_factor = img_crp_resized.shape[0] / img_crp.size[0]
            boxes = []

            for b in range(len(row['x0'])):

                # Adjust boxes' coordinates wrt cropped image:
                x0_new = row['x0'][b] - expanded_square_bb_x0
                y0_new = row['y0'][b] - expanded_square_bb_y0
                x1_new = row['x1'][b] - expanded_square_bb_x0
                y1_new = row['y1'][b] - expanded_square_bb_y0

                # Adjust boxes' coordinates wrt cropped and resized image:
                x0_new = round(x0_new * scaling_factor)
                y0_new = round(y0_new * scaling_factor)
                x1_new = round(x1_new * scaling_factor)
                y1_new = round(y1_new * scaling_factor)

                boxes.append([x0_new, y0_new, x1_new, y1_new])

            self.boxes_list.append(boxes)

            lbls = row['label']

            if self.simplified:
                # All boxes will have label 1 in simplified version, instead of digit labels
                lbls = [1] * len(lbls)

            self.lbls_list.append(lbls)

            total_num_of_processed_files = total_num_of_processed_files + 1

        # Save pickle file in memory
        pickle.dump((self.img_list, self.boxes_list, self.lbls_list), open(self.pkl_file, 'wb'))

        print(f'\nTotal number of processed files: {total_num_of_processed_files}\n')

    def __len__(self):
        if self.is_truncated:
            return 1
        return len(self.img_list)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError

        if self.is_truncated:
            index = 0

        if torch.is_tensor(index):
            index = index.tolist()  # type: ignore

        img = self.img_list[index]
        boxes = self.boxes_list[index]
        lbls = self.lbls_list[index]

        img = self.__normalize_image(img).astype(np.float32)

        if self.transform is not None:
            img = self.transform(img)

            # Normalize boxes:
            boxes = [[box_coord / self.resize_size[0] for box_coord in box] for box in boxes]
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(lbls, dtype=torch.int64)

        return img, (boxes, labels)

    @staticmethod
    def collate_fn(batch):
        """
        Since each image may have a different number of objects, we need a collate function
        (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes and labels
        """
        images = []
        boxes_and_labels = []

        for b in batch:
            images.append(b[0])
            boxes_and_labels.append(b[1])

        images = torch.stack(images, dim=0)
        return images, boxes_and_labels

    @staticmethod
    def get_name(index, hdf5_data):
        """Retrieve name field from hdf5 data"""
        name = hdf5_data['/digitStruct/name']
        return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]][()]])

    @staticmethod
    def get_bbox(index, hdf5_data):
        """Retrieve bounding box field from hdf5 data"""
        attrs = pd.DataFrame()
        item = hdf5_data['digitStruct/bbox'][index].item()
        for key in ['label', 'left', 'top', 'width', 'height']:
            attr = hdf5_data[item][key]
            values = [int(hdf5_data[attr[()][i].item()][()][0][0])
                      for i in range(len(attr))] if len(attr) > 1 else [attr[()][0][0]]
            attrs[key] = values
        # Rename 'left' to 'x0'
        attrs['x0'] = attrs.pop('left')
        # Rename 'top' to 'y0'
        attrs['y0'] = attrs.pop('top')

        return attrs

    @staticmethod
    def read_digit_mat(mat_file):
        """Reading digit information from a .mat file"""
        f = h5py.File(mat_file, 'r')

        info_df = pd.DataFrame()

        bbox = '/digitStruct/bbox'
        for j in range(f[bbox].shape[0]):  # type: ignore # pylint: disable=no-member
            img_name = SVHN.get_name(j, f)
            row_dict = SVHN.get_bbox(j, f)

            row_dict['img_name'] = img_name
            row_dict['x1'] = row_dict['x0'] + row_dict['width']
            row_dict['y1'] = row_dict['y0'] + row_dict['height']

            info_df = pd.concat([info_df, row_dict])

        return info_df

    @staticmethod
    def get_image_size(image_path):
        """Returns image dimensions
        """
        image = Image.open(image_path)
        return image.size

    @staticmethod
    def __normalize_image(image):
        """Normalizes RGB images
        """
        return image / 256

    @staticmethod
    def __makedir_exist_ok(dirpath):
        """Make directory if not already exists
        """
        try:
            os.makedirs(dirpath)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

    @staticmethod
    def fold_image(img, fold_ratio):
        """Folds high resolution H-W-3 image h-w-c such that H * W * 3 = h * w * c.
           These correspond to c/3 downsampled images of the original high resolution image."""
        if fold_ratio == 1:
            img_folded = img
        else:
            img_folded = None
            for i in range(fold_ratio):
                for j in range(fold_ratio):
                    if img_folded is not None:
                        img_folded = \
                            np.concatenate((img_folded, img[i::fold_ratio,
                                                            j::fold_ratio, :]), axis=2)
                    else:
                        img_folded = img[i::fold_ratio, j::fold_ratio, :]
        return img_folded


def SVHN_get_datasets(data, load_train=True, load_test=True, resize_size=(96, 96), fold_ratio=2,
                      simplified=False):

    """ Returns SVHN Dataset
    """
    (data_dir, args) = data

    if load_train:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = SVHN(root_dir=data_dir, d_type='train',
                             transform=train_transform, resize_size=resize_size,
                             fold_ratio=fold_ratio, simplified=simplified)
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             ai8x.normalize(args=args)])

        test_dataset = SVHN(root_dir=data_dir, d_type='test',
                            transform=test_transform, resize_size=resize_size,
                            fold_ratio=fold_ratio, simplified=simplified)
    else:
        test_dataset = None

    return train_dataset, test_dataset


def SVHN_74_get_datasets(data, load_train=True, load_test=True):
    """ Returns SVHN Dataset with 96x96 images
    """
    return SVHN_get_datasets(data, load_train, load_test, resize_size=(74, 74), fold_ratio=1)


def SVHN_74_simplified_get_datasets(data, load_train=True, load_test=True):
    """ Returns SVHN Dataset with 96x96 images and simplified labels: 1 for every digit
    """
    return SVHN_get_datasets(data, load_train, load_test, resize_size=(74, 74), fold_ratio=1,
                             simplified=True)


datasets = [
   {
       'name': 'SVHN_74',
       'input': (3, 74, 74),
       'output': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
       'loader': SVHN_74_get_datasets,
       'collate': SVHN.collate_fn
   },
   {
       'name': 'SVHN_74_simplified',
       'input': (3, 74, 74),
       'output': ([1]),
       'loader': SVHN_74_simplified_get_datasets,
       'collate': SVHN.collate_fn
   }
]
