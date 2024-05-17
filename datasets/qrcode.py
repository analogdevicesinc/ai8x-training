###################################################################################################
#
# Copyright (C) 2024 Analog Devices, Inc. All Rights Reserved.
# This software is proprietary to Analog Devices, Inc. and its licensors.
#
###################################################################################################
"""
Main classes and functions to create a new dataset from randomly generated QR codes.
"""

import random
from urllib.request import Request, urlopen

import numpy as np
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms

import albumentations as album
import cv2
import qrcode

import ai8x
from datasets.bg20k import BG20K
from datasets.image_mixer import (ImageMixerWithObjBBox, ImageMixerWithObjBBoxKeyPts,
                                  ImageMixerWithObjSegment)
from utils import object_detection_utils


class RandomWordGenerator():
    """
    Class to generate random English words
    """
    def __init__(self):
        url = "https://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        web_byte = urlopen(req).read()
        long_txt = web_byte.decode('utf-8')
        self.words = long_txt.splitlines()

    def get_random_word(self):
        """
        Returns random word
        """
        idx = random.randint(0, len(self.words)-1)
        return self.words[idx]


class QRCodeGenerator(Dataset):
    """
    Class to populate a dataset that includes random QR codes with different geometric
    and chromatic transformations.
    """
    min_qr_size = 2
    max_qr_size = 10
    min_num_words = 1
    max_num_words = 4

    def __init__(self, root_dir, d_type, data_len,  # pylint: disable=unused-argument
                 transform=None, augment_data=False, segment_out=False, keypoint_out=False):
        self.data_len = data_len
        self.transform = transform

        self.random_text_list = []
        self.augment_data = augment_data
        self.segment_out = segment_out
        self.keypoint_out = keypoint_out

        self.__gen_random_words()

        # define geometric transforms
        self.g_transforms = \
            album.Compose([album.Affine(scale=(0.6, 1.),
                                        translate_percent=(0.2, 0.4),
                                        rotate=(-45, 45),
                                        mode=cv2.BORDER_CONSTANT,
                                        fit_output=True,
                                        p=0.9),
                           album.Perspective(scale=(0.05, 0.2), p=0.5)],
                          bbox_params=album.BboxParams(format='pascal_voc',
                                                       label_fields=['class_labels']),
                          keypoint_params=album.KeypointParams(format='xy',
                                                               remove_invisible=False))
        # define chromatic transforms
        self.c_transforms = \
            album.Compose([album.RGBShift(r_shift_limit=64, g_shift_limit=64,
                                          b_shift_limit=64, p=0.9),
                           album.ColorJitter(brightness=0.5, contrast=0.5,
                                             saturation=0.5, hue=0.5, p=0.9),
                           album.MultiplicativeNoise(multiplier=(0.5, 1.5), per_channel=True,
                                                     elementwise=True, p=0.7),
                           album.MotionBlur(p=0.7)],
                          bbox_params=album.BboxParams(format='pascal_voc',
                                                       label_fields=['class_labels']),
                          keypoint_params=album.KeypointParams(format='xy',
                                                               remove_invisible=False))

    def __gen_random_words(self):
        random_word_gen = RandomWordGenerator()

        for _ in range(self.data_len):
            num_words = random.randint(self.min_num_words, self.max_num_words)

            text = random_word_gen.get_random_word()
            for _ in range(num_words-1):
                text = ' '.join([text, random_word_gen.get_random_word()])
            self.random_text_list.append(text)

    def __gen_qr(self, text):
        qr_size = random.randint(self.min_qr_size, self.max_qr_size)
        border_size = max(2, int(qr_size / 5))

        qr = qrcode.QRCode(version=1,
                           error_correction=qrcode.constants.ERROR_CORRECT_L,
                           box_size=qr_size,
                           border=border_size)

        qr.add_data(text)
        qr.make(fit=True)

        return qr.make_image(fill_color="black", back_color="white")

    def __clamp_kpts(self, keypoints, img_width, img_height):
        clamp_kpts = keypoints

        for idx1, kpt in enumerate(keypoints):
            clamp_kpts[idx1] = (min(max(0, kpt[0]), img_width-1),
                                min(max(0, kpt[1]), img_height-1))

        return clamp_kpts

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        text = self.random_text_list[index]
        qr_image = self.__gen_qr(text)
        qr_image = 245 * np.asarray(qr_image).astype(np.uint8) + 10

        image = qr_image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        start_x = start_y = 0
        boxes = [[start_x,
                  start_y,
                  start_x + qr_image.shape[1]-1-2,
                  start_y+qr_image.shape[0]-1-2]]
        keypoints = [(start_x, start_y),
                     (start_x + qr_image.shape[1]-1-2, start_y),
                     (start_x, start_y+qr_image.shape[0]-1-2),
                     (start_x + qr_image.shape[1]-1-2, start_y+qr_image.shape[0]-1-2)]
        labels = [1]
        gt_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.int64)

        if self.augment_data:
            transform_out = self.g_transforms(image=image, bboxes=boxes,
                                              keypoints=keypoints, class_labels=labels)
            image = transform_out['image']
            zero_pixels = np.all(image == 0, axis=2)
            if self.segment_out:
                if gt_map.shape != image.shape[:2]:
                    gt_map = np.ones((image.shape[0], image.shape[1]), dtype=np.int64)
                    gt_map[zero_pixels] = 0

            transform_out = self.c_transforms(image=image,
                                              bboxes=transform_out['bboxes'],
                                              keypoints=transform_out['keypoints'],
                                              class_labels=transform_out['class_labels'])
            image = transform_out['image']
            image[zero_pixels] = 0
            boxes = transform_out['bboxes']
            keypoints = transform_out['keypoints']
            keypoints = self.__clamp_kpts(keypoints, image.shape[1], image.shape[0])
            labels = transform_out['class_labels']

        if self.transform is not None:
            image = self.transform(image)

        if self.segment_out:
            return image, gt_map
        if self.keypoint_out:
            return image, (boxes, keypoints, labels)

        return image, (boxes, labels)


def qrcode_get_datasets(data, load_train=True, load_test=True, im_size=(320, 240),
                        fg_to_bg_ratio_range=(0.1, 0.7), num_qr_per_img=1):
    """
    Returns QR Dataset with qr codes' bounding boxes as ground truths
    """
    (data_dir, args) = data

    train_dataset = test_dataset = None
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         ai8x.normalize(args=args)])

    if load_train:
        bg_dataset = BG20K(root_dir=data_dir, d_type='train', transform=None)
        fg_dataset = []
        for _ in range(num_qr_per_img):
            fg_dataset.append(QRCodeGenerator(root_dir=data_dir, d_type='train',
                                              data_len=10000, augment_data=True))

        train_dataset = ImageMixerWithObjBBox('train', fg_dataset, bg_dataset,
                                              transform=data_transform, resize_size=im_size,
                                              fg_to_bg_ratio_range=fg_to_bg_ratio_range)

    if load_test:
        bg_dataset = BG20K(root_dir=data_dir, d_type='test', transform=None)
        fg_dataset = []
        for _ in range(num_qr_per_img):
            fg_dataset.append(QRCodeGenerator(root_dir=data_dir, d_type='train',
                                              data_len=2000, augment_data=True))

        test_dataset = ImageMixerWithObjBBox('test', fg_dataset, bg_dataset,
                                             transform=data_transform, resize_size=im_size,
                                             fg_to_bg_ratio_range=fg_to_bg_ratio_range)

    return train_dataset, test_dataset


def qrcode_get_segmentation_datasets(data, load_train=True, load_test=True, im_size=(352, 352),
                                     data_len=10000, fg_to_bg_ratio_range=(0.1, 0.7),
                                     num_qr_per_img=1):
    """
    Returns QR Dataset with QR codes' segments as ground truths
    """
    (data_dir, args) = data

    train_dataset = test_dataset = None
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         ai8x.normalize(args=args),
                                         ai8x.fold(fold_ratio=4)])

    if load_train:
        bg_dataset = BG20K(root_dir=data_dir, d_type='train', transform=None)
        fg_dataset = []
        for _ in range(num_qr_per_img):
            fg_dataset.append(QRCodeGenerator(root_dir=data_dir, d_type='train',
                                              data_len=data_len, augment_data=True,
                                              segment_out=True))

        train_dataset = ImageMixerWithObjSegment('train', fg_dataset, bg_dataset,
                                                 transform=data_transform, resize_size=im_size,
                                                 fg_to_bg_ratio_range=fg_to_bg_ratio_range)

    if load_test:
        bg_dataset = BG20K(root_dir=data_dir, d_type='test', transform=None)
        fg_dataset = []
        for _ in range(num_qr_per_img):
            fg_dataset.append(QRCodeGenerator(root_dir=data_dir, d_type='train',
                                              data_len=(data_len // 4),
                                              augment_data=True, segment_out=True))

        test_dataset = ImageMixerWithObjSegment('test', fg_dataset, bg_dataset,
                                                transform=data_transform, resize_size=im_size,
                                                fg_to_bg_ratio_range=fg_to_bg_ratio_range)

    return train_dataset, test_dataset


def qrcode_get_keypoint_datasets(data, load_train=True, load_test=True, im_size=(320, 240),
                                 data_len=10000, fg_to_bg_ratio_range=(0.1, 0.7),
                                 num_qr_per_img=1):
    """
    Returns QR Dataset with qr codes' bounding boxes and keypoints as ground truths
    """
    (data_dir, args) = data

    train_dataset = test_dataset = None
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         ai8x.normalize(args=args)])

    if load_train:
        bg_dataset = BG20K(root_dir=data_dir, d_type='train', transform=None)
        fg_dataset = []
        for _ in range(num_qr_per_img):
            fg_dataset.append(QRCodeGenerator(root_dir=data_dir, d_type='train',
                                              data_len=data_len, augment_data=True,
                                              segment_out=False, keypoint_out=True))

        train_dataset = ImageMixerWithObjBBoxKeyPts('train', fg_dataset, bg_dataset,
                                                    transform=data_transform,
                                                    resize_size=im_size,
                                                    fg_to_bg_ratio_range=fg_to_bg_ratio_range)

    if load_test:
        bg_dataset = BG20K(root_dir=data_dir, d_type='test', transform=None)
        fg_dataset = []
        for _ in range(num_qr_per_img):
            fg_dataset.append(QRCodeGenerator(root_dir=data_dir, d_type='train',
                                              data_len=(data_len // 4), augment_data=True,
                                              segment_out=False, keypoint_out=True))

        test_dataset = ImageMixerWithObjBBoxKeyPts('test', fg_dataset, bg_dataset,
                                                   transform=data_transform, resize_size=im_size,
                                                   fg_to_bg_ratio_range=fg_to_bg_ratio_range)

    return train_dataset, test_dataset


def qrcode_get_datasets_qqvga(data, load_train=True, load_test=True):
    """Returns QRCode datasets in qqVGA (160x120) resolution"""
    return qrcode_get_datasets(data, load_train, load_test, im_size=(160, 120),
                               fg_to_bg_ratio_range=(0.05, 0.95))


def qrcode_get_kpts_datasets_qqvga(data, load_train=True, load_test=True):
    """Returns QRCode datasets in qqVGA (160x120) resolution"""
    return qrcode_get_keypoint_datasets(data, load_train, load_test, im_size=(160, 120),
                                        fg_to_bg_ratio_range=(0.05, 0.95))


def qrcode_get_kpts_ext_datasets_qqvga(data, load_train=True, load_test=True):
    """Returns QRCode datasets in qqVGA (160x120) resolution"""
    train_set1, test_set1 = qrcode_get_keypoint_datasets(data, load_train, load_test,
                                                         im_size=(160, 120), data_len=10000,
                                                         fg_to_bg_ratio_range=(0.05, 0.95))
    train_set2, test_set2 = qrcode_get_keypoint_datasets(data, load_train, load_test,
                                                         im_size=(160, 120), data_len=3000,
                                                         fg_to_bg_ratio_range=(0.75, 0.99))

    train_set = test_set = None
    if load_train:
        train_set = ConcatDataset([train_set1, train_set2])

    if load_test:
        test_set = ConcatDataset([test_set1, test_set2])

    return train_set, test_set


datasets = [
    {
        'name': 'qrcode_160_120',
        'input': (3, 120, 160),
        'output': ([1]),
        'loader': qrcode_get_datasets_qqvga,
        'collate': object_detection_utils.collate_fn
    },
    {
        'name': 'qrcode_160_120_kpts',
        'input': (3, 120, 160),
        'output': ([1]),
        'loader': qrcode_get_kpts_datasets_qqvga,
        'collate': object_detection_utils.collate_fn
    },
    {
        'name': 'qrcode_160_120_kpts_ext',
        'input': (3, 120, 160),
        'output': ([1]),
        'loader': qrcode_get_kpts_ext_datasets_qqvga,
        'collate': object_detection_utils.collate_fn
    },
]
