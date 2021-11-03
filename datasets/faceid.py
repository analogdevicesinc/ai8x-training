###################################################################################################
#
# Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Classes and functions used to utilize the Face ID dataset.
"""
import os

from torchvision import transforms

import ai8x
from datasets.vggface2 import VGGFace2Dataset
from datasets.youtube_faces import YouTubeFacesDataset


def faceid_get_datasets(data, load_train=True, load_test=True):
    """
    Load the faceID dataset

    The dataset is loaded from the archive file, so the file is required for this version.

    The dataset consists of actually 2 different datasets, VGGFace2 for training and YouTubeFaces
    for the test. The reason of this is proof-of-concept models are obtained by this way and the
    losses At YTFaces are tracked for the sake of benchmarking.

    The images are all 3-color 160x120 sized and consist the face image.
    """
    (data_dir, args) = data

    # These are hard coded for now, need to come from above in future.
    train_resample_subj = 1
    train_resample_img_per_subj = 6
    test_resample_subj = 1
    test_resample_img_per_subj = 2
    train_data_dir = os.path.join(data_dir, 'VGGFace-2')
    test_data_dir = os.path.join(data_dir, 'YouTubeFaces')

    transform = transforms.Compose([
        ai8x.normalize(args=args)
    ])

    if load_train:
        train_dataset = VGGFace2Dataset(root_dir=train_data_dir, d_type='train',
                                        transform=transform,
                                        resample_subj=train_resample_subj,
                                        resample_img_per_subj=train_resample_img_per_subj)
    else:
        train_dataset = None

    if load_test:
        test_dataset = YouTubeFacesDataset(root_dir=test_data_dir, d_type='test',
                                           transform=transform,
                                           resample_subj=test_resample_subj,
                                           resample_img_per_subj=test_resample_img_per_subj)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'FaceID',
        'input': (3, 160, 120),
        'output': ('id'),
        'regression': True,
        'loader': faceid_get_datasets,
    },
]
