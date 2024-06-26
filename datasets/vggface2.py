###################################################################################################
#
# Copyright (C) 2019-2024 Maxim Integrated Products, Inc. All Rights Reserved.
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

import numpy as np
import torch
import torchvision.transforms.functional as FT
from torch.utils.data import Dataset
from torchvision import transforms

import cv2
import kornia.geometry.transform as GT
from batch_face import RetinaFace
from PIL import Image
from skimage import transform as trans
from tqdm import tqdm

import ai8x
from utils import augmentation_utils


class VGGFace2(Dataset):
    """
    VGGFace2 Dataset
    """
    def __init__(self, root_dir, d_type, mode, transform=None,
                 teacher_transform=None, img_size=(112, 112), args=None):

        if d_type not in ('test', 'train'):
            raise ValueError("d_type can only be set to 'test' or 'train'")

        if mode not in ('detection', 'identification', 'identification_dr'):
            raise ValueError("mode can only be set to 'detection', 'identification',"
                             "or 'identification_dr'")

        self.device = args.device
        self.root_dir = root_dir
        self.d_type = d_type
        self.transform = transform
        self.teacher_transform = teacher_transform
        self.img_size = img_size
        self.mode = mode
        self.dataset_path = os.path.join(self.root_dir, "VGGFace-2")
        self.__makedir_exist_ok(self.dataset_path)
        self.count = 0
        self.tform = trans.SimilarityTransform()
        self.src = np.array([
                            [38.2946, 51.6963],
                            [73.5318, 51.5014],
                            [56.0252, 71.7366],
                            [41.5493, 92.3655],
                            [70.7299, 92.2041]], dtype=np.float32)

        self.__makedir_exist_ok(self.dataset_path)
        self.__makedir_exist_ok(os.path.join(self.dataset_path, "processed"))

        if self.d_type in ('train', 'test'):
            self.gt_path = os.path.join(self.dataset_path, "processed",
                                        self.d_type+"_vggface2.pickle")
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

            f = open(self.gt_path, 'rb')
            self.pickle_dict = pickle.load(f)
            f.close()

        else:
            print(f'Unknown data type: {self.d_type}')
            return

    def __extract_gt(self):
        """
        Extracts the ground truth from the dataset
        """
        if self.device == 'cuda':
            detector = RetinaFace(gpu_id=torch.cuda.current_device(), network="resnet50")
        else:
            detector = RetinaFace(gpu_id=-1, network="resnet50")

        img_paths = list(glob.glob(os.path.join(self.d_path + '/**/', '*.jpg'), recursive=True))
        nf_number = 0
        words_count = 0
        pickle_dict = {key: [] for key in ["boxes", "landmarks", "img_list", "lbl_list"]}
        pickle_dict["word2index"] = {}

        for jpg in tqdm(img_paths):
            boxes = []
            image = cv2.imread(jpg)

            faces = detector(image)

            if len(faces) == 0:
                nf_number += 1
                continue

            for face in faces:
                box = face[0]
                box = np.clip(box[:4], 0, None)
                boxes.append(box)
            lndmrks = faces[0][1]

            dir_name = os.path.dirname(jpg)
            lbl = os.path.relpath(dir_name, self.d_path)

            if lbl not in pickle_dict["word2index"]:
                pickle_dict["word2index"][lbl] = words_count
                words_count += 1

            pickle_dict["lbl_list"].append(lbl)
            pickle_dict["boxes"].append(boxes)
            pickle_dict["landmarks"].append(lndmrks)
            pickle_dict["img_list"].append(os.path.relpath(jpg, self.dataset_path))
        if nf_number > 0:
            print(f'Not found any faces in {nf_number} images ')

        with open(self.gt_path, 'wb') as f:
            pickle.dump(pickle_dict, f)

    def __len__(self):
        return len(self.pickle_dict["img_list"]) - 1

    def __getitem__(self, index):
        """
        Get the image and associated target according to the mode
        """
        if index >= len(self):
            raise IndexError

        if self.mode == 'detection':
            return self.__getitem_detection(index)

        if self.mode == 'identification':
            return self.__getitem_identification(index)

        if self.mode == 'identification_dr':
            return self.__getitem_identification_dr(index)

        # Will never reached
        return None

    def __getitem_detection(self, index):
        """
        Get the image and associated target for face detection
        """
        if torch.is_tensor(index):
            index = index.tolist()

        img = Image.open(os.path.join(self.dataset_path, self.pickle_dict["img_list"][index]))
        img = FT.to_tensor(img)

        boxes = self.pickle_dict["boxes"][index]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        img, boxes = augmentation_utils.resize(img, boxes,
                                               dims=(self.img_size[0], self.img_size[1]))

        labels = [1] * boxes.shape[0]

        if self.transform is not None:
            img = self.transform(img)

        boxes = boxes.clamp_(min=0, max=1)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        return img, (boxes, labels)

    def __getitem_identification(self, index):
        """
        Get the image and associated target for face identification
        """
        if torch.is_tensor(index):
            index = index.tolist()

        lbl = self.pickle_dict["lbl_list"][index]
        lbl_index = self.pickle_dict["word2index"][lbl]
        lbl_index = torch.tensor(lbl_index, dtype=torch.long)
        box = self.pickle_dict["boxes"][index][0]
        img = Image.open(os.path.join(self.dataset_path, self.pickle_dict["img_list"][index]))
        img_A = img.copy()

        # Apply transformation to the image that will be aligned
        if self.teacher_transform is not None:
            img_A = self.teacher_transform(img_A)

        # Apply transformation to the image that will be cropped
        if self.transform is not None:
            img = self.transform(img)

        # Use landmarks to estimate affine transformation
        landmark = self.pickle_dict["landmarks"][index]
        self.tform.estimate(landmark, self.src)
        A = self.tform.params[0:2, :]
        A = torch.as_tensor(A, dtype=torch.float32)
        A = A.unsqueeze(0)

        # Apply affine transformation to obtain aligned image
        img_A = GT.warp_affine(img_A.unsqueeze(0), A, (self.img_size[0], self.img_size[1]))
        img_A = img_A.squeeze(0)

        # Convert bounding box to square
        height = box[3] - box[1]
        width = box[2] - box[0]
        max_dim = max(height, width)
        box[0] = np.clip(box[0] - (max_dim - width) / 2, 0, img.shape[2])
        box[1] = np.clip(box[1] - (max_dim - height) / 2, 0, img.shape[1])
        box[2] = np.clip(box[2] + (max_dim - width) / 2, 0, img.shape[2])
        box[3] = np.clip(box[3] + (max_dim - height) / 2, 0, img.shape[1])

        # Crop image with the square bounding box
        img_C = FT.crop(img=img, top=int(box[1]), left=int(box[0]),
                        height=int(box[3]-box[1]), width=int(box[2]-box[0]))

        # Check if the cropped image is square, if not, pad it
        _, h, w = img_C.shape
        if w != h:
            max_dim = max(w, h)
            h_padding = (max_dim - h) / 2
            w_padding = (max_dim - w) / 2
            l_pad = w_padding if w_padding % 1 == 0 else w_padding+0.5
            t_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
            r_pad = w_padding if w_padding % 1 == 0 else w_padding-0.5
            b_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
            padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
            img_C = FT.pad(img_C, padding, 0, 'constant')

        # Resize cropped image to the desired size
        img_C = FT.resize(img_C, (self.img_size[0], self.img_size[1]))

        # Concatenate images
        concat_img = torch.cat((img_C, img_A), 0)

        return concat_img, lbl_index

    def __getitem_identification_dr(self, index):
        """
        Get the image and associated target for dimensionality reduction
        """
        if torch.is_tensor(index):
            index = index.tolist()

        lbl = self.pickle_dict["lbl_list"][index]
        lbl_index = self.pickle_dict["word2index"][lbl]
        lbl_index = torch.tensor(lbl_index, dtype=torch.long)
        img = Image.open(os.path.join(self.dataset_path, self.pickle_dict["img_list"][index]))

        # Apply transformation to the image that will be aligned
        if self.transform is not None:
            img = self.transform(img)

        # Use landmarks to estimate affine transformation
        landmark = self.pickle_dict["landmarks"][index]
        self.tform.estimate(landmark, self.src)
        A = self.tform.params[0:2, :]
        A = torch.as_tensor(A, dtype=torch.float32)
        A = A.unsqueeze(0)

        # Apply affine transformation to obtain aligned image
        img = GT.warp_affine(img.unsqueeze(0), A, (self.img_size[0], self.img_size[1]))
        img = img.squeeze(0)

        return img, lbl_index

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


def VGGFace2_FaceID_get_datasets(data, load_train=True, load_test=True, img_size=(112, 112)):

    """ Returns FaceID Dataset
    """
    (data_dir, args) = data

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=(0.6, 1.4), saturation=(0.6, 1.4),
                               contrast=(0.6, 1.4), hue=(-0.4, 0.4)),
        transforms.RandomErasing(p=0.1),
        ai8x.normalize(args=args)])

    teacher_transform = transforms.Compose([
        transforms.ToTensor(),
        ai8x.normalize(args=args)])

    if load_train:

        train_dataset = VGGFace2(root_dir=data_dir, d_type='train', mode='identification',
                                 transform=train_transform, teacher_transform=teacher_transform,
                                 img_size=img_size, args=args)

        print(f'Train dataset length: {len(train_dataset)}\n')
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([transforms.ToTensor(),
                                            ai8x.normalize(args=args)])

        test_dataset = VGGFace2(root_dir=data_dir, d_type='test', mode='identification',
                                transform=test_transform, teacher_transform=teacher_transform,
                                img_size=img_size, args=args)

        print(f'Test dataset length: {len(test_dataset)}\n')
    else:
        test_dataset = None

    return train_dataset, test_dataset


def VGGFace2_FaceID_dr_get_datasets(data, load_train=True, load_test=True, img_size=(112, 112)):

    """ Returns FaceID Dataset for dimensionality reduction
    """
    (data_dir, args) = data

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        ai8x.normalize(args=args)])

    if load_train:

        train_dataset = VGGFace2(root_dir=data_dir, d_type='train', mode='identification_dr',
                                 transform=train_transform, img_size=img_size, args=args)

        print(f'Train dataset length: {len(train_dataset)}\n')
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([transforms.ToTensor(),
                                            ai8x.normalize(args=args)])

        test_dataset = VGGFace2(root_dir=data_dir, d_type='test', mode='identification_dr',
                                transform=test_transform, img_size=img_size, args=args)

        print(f'Test dataset length: {len(test_dataset)}\n')
    else:
        test_dataset = None

    return train_dataset, test_dataset


def VGGFace2_Facedet_get_datasets(data, load_train=True, load_test=True, img_size=(224, 168)):

    """ Returns FaceDetection Dataset
    """
    (data_dir, args) = data

    if load_train:
        train_transform = transforms.Compose([
            ai8x.normalize(args=args)])

        train_dataset = VGGFace2(root_dir=data_dir, d_type='train', mode='detection',
                                 transform=train_transform, img_size=img_size, args=args)

        print(f'Train dataset length: {len(train_dataset)}\n')
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([ai8x.normalize(args=args)])

        test_dataset = VGGFace2(root_dir=data_dir, d_type='test', mode='detection',
                                transform=test_transform, img_size=img_size, args=args)

        print(f'Test dataset length: {len(test_dataset)}\n')
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
       'name': 'VGGFace2_FaceID',
       'input': (3, 112, 112),
       'output': ('id'),
       'loader': VGGFace2_FaceID_get_datasets,
    },
    {
       'name': 'VGGFace2_FaceID_dr',
       'input': (3, 112, 112),
       'output': [*range(0, 8631, 1)],
       'loader': VGGFace2_FaceID_dr_get_datasets,
    },
    {
       'name': 'VGGFace2_FaceDetection',
       'input': (3, 224, 168),
       'output': ([1]),
       'loader': VGGFace2_Facedet_get_datasets,
       'collate': VGGFace2.collate_fn
    }
]
