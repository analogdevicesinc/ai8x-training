###################################################################################################
#
# Copyright (C) 2019-2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
YouTube Faces Dataset
https://www.cs.tau.ac.il/~wolf/ytfaces/
"""
import os
import pickle
import time

import numpy as np
import torch
from torch.utils import data


class YouTubeFacesDataset(data.Dataset):
    """
    YouTube Faces Dataset
    https://www.cs.tau.ac.il/~wolf/ytfaces/
    """
    def __init__(
            self,
            root_dir,
            d_type,
            transform=None,
            resample_subj=1,
            resample_img_per_subj=1,
    ):
        data_folder = os.path.join(root_dir, d_type)
        assert os.path.isdir(data_folder), (f'No dataset at {data_folder}.'
                                            ' Follow the steps at datasets/face_id/README.md')

        data_file_list = sorted([d for d in os.listdir(data_folder) if d.startswith('whole_set')])

        self.sid_list = []
        self.embedding_list = []
        self.img_list = []
        self.transform = transform

        subj_idx = 0
        n_elems = 0

        t_start = time.time()
        print('Data loading...')
        for n_file, data_file in enumerate(data_file_list):
            print(f'\t{n_file+1} of {len(data_file_list)}')
            f_path = os.path.join(data_folder, data_file)

            with open(f_path, 'rb') as f:
                x = pickle.load(f)

            for key in list(x)[::resample_subj]:
                val = x[key]
                for key2 in list(val)[::resample_img_per_subj]:
                    for key3 in list(val[key2]):
                        self.img_list.append(val[key2][key3]['img'])
                        self.embedding_list.append(
                            np.array(val[key2][key3]['embedding']).astype(np.float32)
                        )
                        self.sid_list.append(subj_idx)
                        n_elems += 1
                subj_idx += resample_subj

        t_end = time.time()
        print(f'{n_elems} of data samples loaded in {t_end-t_start:.4f} seconds.')

    def __normalize_data(self, data_item):
        data_item = data_item.astype(np.float32)
        data_item /= 256
        return data_item

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        embedding = self.embedding_list[idx]
        embedding = np.expand_dims(embedding, 1)
        embedding = np.expand_dims(embedding, 2)
        embedding *= 6.0

        inp = torch.tensor(self.__normalize_data(self.img_list[idx]), dtype=torch.float)
        if self.transform is not None:
            inp = self.transform(inp)

        return inp, torch.tensor(embedding, dtype=torch.float)
