###################################################################################################
#
# Copyright (C) 2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Classes and functions used to create Pascal VOC Datasets (http://host.robots.ox.ac.uk/pascal/VOC/)
PyTorch Torch Vision Dataset
https://pytorch.org/vision/main/generated/torchvision.datasets.VOCDetection.html
"""
import os
import random

import torch
import torchvision
import torchvision.transforms.functional as FT
from torchvision import transforms

import ai8x
from utils import augmentation_utils, object_detection_utils


class PascalVOC(torch.utils.data.Dataset):
    """
    The PASCAL Visual Object Classes (http://host.robots.ox.ac.uk/pascal/VOC/)
    Object detection challenge dataset is used.
    Torch metrics implementation is encapsulated.
    (https://pytorch.org/vision/main/generated/torchvision.datasets.VOCDetection.html)
    """

    year_options = {'min': 2007, 'max': 2012}

    # For years 2008 and above, supported classes are common for all:
    voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                  'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                  'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                  'train', 'tvmonitor')

    voc_label_to_id_map = {k: v + 1 for v, k in enumerate(voc_labels)}
    voc_id_to_label_map = {v: k for k, v in voc_label_to_id_map.items()}

    def __init__(self, root, d_type, img_size, year_opt, augment_data,
                 transform, download=False):

        self.data_root = os.path.join(root, self.__class__.__name__)
        self.img_size = img_size
        self.augment_data = augment_data
        self.transform = transform

        self.__check_for_download(year_opt, d_type, download)
        self.__prepare_dataset(d_type)

    def __check_for_download(self, year_opt, d_type, download):
        self.year_list = []
        self.download = {}

        year_opt_list = year_opt.split('_')
        for year_str in year_opt_list:

            if int(year_str) >= self.year_options['min'] and \
               int(year_str) <= self.year_options['max']:

                self.year_list.append(year_str)
                self.download[year_str] = download and \
                    (not self.__check_data_path_exists(year_str, d_type))
            else:
                print(f'Warning: {year_str} is not a valid year for Pascal VOC!')

    def __check_data_path_exists(self, year_str, d_type):
        return os.path.exists(os.path.join(self.data_root, 'VOCdevkit',
                                           'VOC' + year_str, 'ImageSets',
                                           'Main', d_type + '.txt'))

    def __prepare_dataset(self, dtype):
        self.dataset = None

        for year_str in self.year_list:
            dataset_temp = \
                torchvision.datasets.VOCDetection(root=self.data_root, year=year_str,
                                                  image_set=dtype,
                                                  download=self.download[year_str],
                                                  transforms=self.transforms_func)

            if self.dataset:
                self.dataset = torch.utils.data.ConcatDataset([self.dataset, dataset_temp])
            else:
                self.dataset = dataset_temp

    def transforms_func(self, image, target):
        """
        Torch vision VOCDetection dataset's __get_item__ returns: (image, target)
        where target is a dictionary of the XML tree.
        This transform function parses boxes and labels from the target file.
        This transform function also applies transformations on images, boxes and
        labels as follows:
            * images: are normalized (torchvision.transforms.ToTensor performs
                      normalization while transforming PIL image into Tensor)
                      are resized
            * boxes : are normalized, resized wrt resized image
            * labels: are mapped to label id/s (integer)
        """

        boxes = []
        labels = []
        difficulties = []

        for obj in target['annotation']['object']:

            label = obj['name']
            box = [int(float(obj['bndbox']['xmin'])), int(float(obj['bndbox']['ymin'])),
                   int(float(obj['bndbox']['xmax'])), int(float(obj['bndbox']['ymax']))]

            difficulty = int(obj['difficult'])

            labels.append(label)
            boxes.append(box)
            difficulties.append(difficulty)

        # labels to label ids:
        labels = [self.voc_label_to_id_map[label] for label in labels]

        does_have_box = len(boxes) != 0

        boxes = torch.tensor(boxes, dtype=torch.float)  # (n_objects, 4)
        labels = torch.tensor(labels, dtype=torch.long)  # (n_objects)
        difficulties = torch.ByteTensor(difficulties)  # (n_objects)

        new_image = image
        new_boxes = boxes
        new_labels = labels

        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(new_image)

        if self.augment_data and does_have_box:
            mean = [0.5, 0.5, 0.5]

            # A series of photometric distortions in random order,
            # each with 50% chance of occurrence, as in Caffe repo
            new_image = augmentation_utils.photometric_distort(new_image)

            # Expand image (zoom out) with a 50% chance - helpful for training
            # detection of small objects
            # Fill surrounding space with the mean of ImageNet data that our base
            # VGG was trained on
            if random.random() < 0.5:
                new_image, new_boxes = \
                    augmentation_utils.expand(new_image, new_boxes, filler=mean)

            # Randomly crop image (zoom in)
            new_image, new_boxes, new_labels, _ =\
                augmentation_utils.random_crop(new_image, new_boxes, new_labels,
                                               difficulties)

            # Flip image with a 50% chance
            if random.random() < 0.5:
                new_image, new_boxes = augmentation_utils.flip(new_image, new_boxes)

        image, boxes = augmentation_utils.resize(new_image, new_boxes, dims=self.img_size)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(new_labels, dtype=torch.long)

        # TODO append difficulties for future use, modify train.py accordingly
        return image, (boxes, labels)

    def __len__(self):
        return len(self.dataset)  # type: ignore # dataset guaranteed to not be None

    def __getitem__(self, index):
        image, target = self.dataset[index]  # type: ignore # dataset guaranteed to not be None
        image = self.transform(image)
        return image, target


def pascal_voc_get_datasets(data, load_train=True, load_test=True, img_size=(300, 300),
                            year_opt='2007', augment_data=True):
    """
    Returns Pascal VOC dataset for the year specified by year_str
    """
    (data_dir, args) = data

    # Apply ai8x normalization and transformations for img:
    transform = transforms.Compose([ai8x.normalize(args=args)])

    if load_train:
        d_type = 'train'
        if '2007' in year_opt:
            d_type = 'trainval'

        train_dataset = PascalVOC(data_dir, d_type, img_size, year_opt,
                                  augment_data=augment_data,
                                  transform=transform,
                                  download=True)
    else:
        train_dataset = None

    if load_test:
        d_type = 'val'
        if '2007' in year_opt:
            d_type = 'test'
            year_opt = '2007'

        test_dataset = PascalVOC(data_dir, d_type, img_size, year_opt,
                                 augment_data=False,
                                 transform=transform, download=True)

        if args.truncate_testset:
            test_dataset.data = \
                test_dataset.data[:1]  # pylint: disable=attribute-defined-outside-init
    else:
        test_dataset = None

    return train_dataset, test_dataset


def pascal_voc_2007_2012_256_320_aug_get_dataset(data, load_train=True,
                                                 load_test=True):
    """ Returns Pascal VOC 2007 and 2012 merged dataset group with augmentation in
        resolution 256x320
    """
    return pascal_voc_get_datasets(data, load_train, load_test,
                                   img_size=(256, 320), year_opt='2007_2012',
                                   augment_data=True)


datasets = [
    {
        'name': 'PascalVOC_2007_2012_256_320_augmented',
        'input': (3, 256, 320),
        'output': PascalVOC.voc_labels,
        'loader': pascal_voc_2007_2012_256_320_aug_get_dataset,
        'collate': object_detection_utils.collate_fn
    },
]
