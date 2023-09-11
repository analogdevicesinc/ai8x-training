###################################################################################################
#
# Copyright (C) 2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
VGGFace2: A Dataset for Recognising Faces across Pose and Age
https://ieeexplore.ieee.org/abstract/document/8373813
"""


import errno
import glob
import os
import pickle

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from tqdm import tqdm

import ai8x
from datasets.face_id.facenet_pytorch import MTCNN


class VGGFace2_FaceDetectionDataset(Dataset):
    """
    VGGFace2 Dataset for face detection

    MTCNN is used to extract the ground truth from the dataset as it provides
    the ground truth for multiple faces in an image.

    GT Format: 0-3:Box Coordinates

    """
    def __init__(self, root_dir, d_type, transform=None, img_size=(224, 168)):

        if d_type not in ('test', 'train'):
            raise ValueError("d_type can only be set to 'test' or 'train'")

        self.root_dir = root_dir
        self.d_type = d_type
        self.transform = transform
        self.img_size = img_size
        self.dataset_path = os.path.join(self.root_dir, "VGGFace-2")
        self.__makedir_exist_ok(self.dataset_path)
        self.__makedir_exist_ok(os.path.join(self.dataset_path, "processed"))

        if self.d_type in ('train', 'test'):
            self.gt_path = os.path.join(self.dataset_path, "processed", self.d_type+"_gt.pickle")
            self.d_path = os.path.join(self.dataset_path, self.d_type)
            if not os.path.exists(self.gt_path):
                assert os.path.isdir(self.d_path), (f'No dataset at {self.d_path}.\n'
                                                    ' Please review the term and'
                                                    ' conditions at https://www.robots.ox.ac.uk/'
                                                    '~vgg/data/vgg_face2/ . Then, download the'
                                                    ' dataset and extract raw images to the'
                                                    ' train and test subfolders.\n'
                                                    ' Expected folder structure: \n'
                                                    ' - root_dir \n'
                                                    '     - VGGFace-2 \n'
                                                    '       - train \n'
                                                    '       - test \n')

                print("Extracting ground truth from the " + self.d_type + " set")
                self.__extract_gt()

        else:
            print(f'Unknown data type: {self.d_type}')
            return

        f = open(self.gt_path, 'rb')
        self.pickle_dict = pickle.load(f)
        f.close()

    def __extract_gt(self):
        """
        Extracts the ground truth from the dataset
        """
        mtcnn = MTCNN()
        img_paths = list(glob.glob(os.path.join(self.d_path + '/**/', '*.jpg'), recursive=True))
        nf_number = 0
        pickle_dict = {key: [] for key in ["gt", "img_list"]}

        for jpg in tqdm(img_paths):
            img = Image.open(jpg)
            img = img.resize((self.img_size[1], self.img_size[0]))
            # pylint: disable-next=unbalanced-tuple-unpacking
            gt, _ = mtcnn.detect(img, landmarks=False)  # type: ignore  # returns tuple of 2

            if gt is None or None in gt:
                nf_number += 1
                continue

            pickle_dict["gt"].append(gt)
            pickle_dict["img_list"].append(os.path.relpath(jpg, self.dataset_path))

        if nf_number > 0:
            print(f'Not found any faces in {nf_number} images ')

        with open(self.gt_path, 'wb') as f:
            pickle.dump(pickle_dict, f)

    def __len__(self):
        return len(self.pickle_dict["img_list"]) - 1

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError

        if torch.is_tensor(index):
            index = index.tolist()

        img = Image.open(os.path.join(self.dataset_path, self.pickle_dict["img_list"][index]))

        ground_truth = self.pickle_dict["gt"][index]

        lbls = [1] * ground_truth.shape[0]
        if self.transform is not None:
            img = self.transform(img)
            for box in ground_truth:
                box[0] = box[0] / self.img_size[1]
                box[2] = box[2] / self.img_size[1]
                box[1] = box[1] / self.img_size[0]
                box[3] = box[3] / self.img_size[0]

            boxes = torch.as_tensor(ground_truth, dtype=torch.float32)
            boxes = boxes.clamp_(min=0, max=1)

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


def VGGFace2_Facedet_get_datasets(data, load_train=True, load_test=True, img_size=(224, 168)):

    """ Returns FaceDetection Dataset
    """
    (data_dir, args) = data

    if load_train:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
            ai8x.normalize(args=args)
        ])

        train_dataset = VGGFace2_FaceDetectionDataset(root_dir=data_dir, d_type='train',
                                                      transform=train_transform, img_size=img_size)

        print(f'Train dataset length: {len(train_dataset)}\n')
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(img_size),
                                            ai8x.normalize(args=args)])

        test_dataset = VGGFace2_FaceDetectionDataset(root_dir=data_dir, d_type='test',
                                                     transform=test_transform, img_size=img_size)

        print(f'Test dataset length: {len(test_dataset)}\n')
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
       'name': 'VGGFace2_FaceDetection',
       'input': (3, 224, 168),
       'output': ([1]),
       'loader': VGGFace2_Facedet_get_datasets,
       'collate': VGGFace2_FaceDetectionDataset.collate_fn
    }
]
