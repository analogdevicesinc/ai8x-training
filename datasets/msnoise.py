#
# Copyright (c) 2018 Intel Corporation
# Portions Copyright (C) 2019-2023 Maxim Integrated Products, Inc.
# Portions Copyright (C) 2023-2024 Analog Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Classes and functions used to create audio noise dataset.
"""
import errno
import json
import os
import sys
import urllib
import urllib.request
import warnings

import numpy as np
import torch
from torchvision import transforms

import librosa

import ai8x


class MSnoise:
    """
    `Microsoft Scalable Noisy Speech <https://github.com/microsoft/MS-SNSD>`
    Dataset, 1D folded.

    Args:
    root (string): Root directory of dataset where ``MSnoise/processed/dataset.pt``
        exist.
    classes(array): List of keywords to be used.
    d_type(string): Option for the created dataset. ``train`` or ``test``.
    dataset_len(int): Dataset length to be returned.
    remove_unknowns (bool, optional): If true, unchosen classes are not gathered as
        the unknown class.
    transform (callable, optional): A function/transform that takes in an PIL image
        and returns a transformed version.
    quantize (bool, optional): If true, the datasets are prepared and saved as
        quantized. If this dataset is to be used with MixedKWS class, then this
        argument must be false.
    download (bool, optional): If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.

    """

    class_dict = {'AirConditioner': 0, 'AirportAnnouncements': 1,
                  'Babble': 2, 'Bus': 3, 'CafeTeria': 4, 'Car': 5,
                  'CopyMachine': 6, 'Field': 7, 'Hallway': 8, 'Kitchen': 9,
                  'LivingRoom': 10, 'Metro': 11, 'Munching': 12, 'NeighborSpeaking': 13,
                  'Office': 14, 'Park': 15, 'Restaurant': 16, 'ShuttingDoor': 17,
                  'Square': 18, 'SqueakyChair': 19, 'Station': 20, 'TradeShow': 21, 'Traffic': 22,
                  'Typing': 23, 'VacuumCleaner': 24, 'WasherDryer': 25, 'Washing': 26}

    def __init__(self, root, classes, d_type, dataset_len, exp_len=16384, fs=16000,
                 noise_time_step=0.25, remove_unknowns=False, transform=None,
                 quantize=False, download=False):
        self.root = root
        self.classes = classes
        self.d_type = d_type
        self.remove_unknowns = remove_unknowns
        self.transform = transform

        self.dataset_len = dataset_len
        self.exp_len = exp_len
        self.fs = fs
        self.noise_time_step = noise_time_step

        self.noise_train_folder = os.path.join(self.raw_folder, 'noise_train')
        self.noise_test_folder = os.path.join(self.raw_folder, 'noise_test')
        self.url_train = \
            'https://api.github.com/repos/microsoft/MS-SNSD/contents/noise_train?ref=master'
        self.url_test = \
            'https://api.github.com/repos/microsoft/MS-SNSD/contents/noise_test?ref=master'
        self.quantize = quantize

        if download:
            self.__download()

        self.data, self.targets, self.data_type, self.rms_val = self.__gen_datasets()

        # rms values for each sample to be returned
        self.rms = np.zeros(self.dataset_len)

        self.__filter_dtype()
        self.__filter_classes()

    @property
    def raw_folder(self):
        """Folder for the raw data.
        """
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    def __download(self):

        if os.path.exists(self.raw_folder):
            return

        self.__makedir_exist_ok(self.noise_train_folder)
        self.__makedir_exist_ok(self.noise_test_folder)

        self.__download_raw(self.url_train)
        self.__download_raw(self.url_test)

    def __download_raw(self, api_url):
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        response = urllib.request.urlretrieve(api_url)

        total_files = 0
        with open(response[0], mode='r', encoding='utf-8') as f:
            data = json.load(f)
            total_files += len(data)
            for file in data:
                file_url = file["download_url"]
                file_name = file["name"]
                path = file["path"]
                path = os.path.join(self.raw_folder, path)
                try:
                    opener = urllib.request.build_opener()
                    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                    urllib.request.install_opener(opener)
                    urllib.request.urlretrieve(file_url, path)
                    print(f'Downloaded: {file_name}')
                except KeyboardInterrupt:
                    print('Interrupted while downloading!')
                    sys.exit()

    def __makedir_exist_ok(self, dirpath):
        try:
            os.makedirs(dirpath)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

    def __filter_dtype(self):

        if self.d_type == 'train':
            bool_list = [i == 0 for i in self.data_type]
            idx_to_select = [i for i, x in enumerate(bool_list) if x]
        elif self.d_type == 'test':
            bool_list = [i == 1 for i in self.data_type]
            idx_to_select = [i for i, x in enumerate(bool_list) if x]
        else:
            print(f'Unknown data type: {self.d_type}')
            return

        self.data = [self.data[i] for i in idx_to_select]
        self.targets = [self.targets[i] for i in idx_to_select]
        self.rms_val = [self.rms_val[i] for i in idx_to_select]
        del self.data_type

    def __filter_classes(self):
        print('\n')
        self.targets = np.array(self.targets)
        initial_new_class_label = len(self.class_dict)
        new_class_label = initial_new_class_label
        for c in self.classes:
            if c not in self.class_dict:
                print(f'Class is not in the data: {c}')
                return
            # else:
            print(f'Class {c}, {self.class_dict[c]}')
            bool_list = [self.class_dict[c] == i for i in self.targets]
            idx = [i for i, x in enumerate(bool_list) if x]
            self.targets[idx] = new_class_label
            print(f'{c}: {new_class_label - initial_new_class_label}')
            new_class_label += 1

        self.targets[(self.targets < initial_new_class_label)] = new_class_label
        if self.remove_unknowns:
            bool_list = [i != new_class_label for i in self.targets]
            idx_to_keep = [i for i, x in enumerate(bool_list) if x]

            self.data = [self.data[i] for i in idx_to_keep]
            self.targets = [self.targets[i] for i in idx]
            self.rms_val = [self.rms_val[i] for i in idx]

        self.targets = [target - initial_new_class_label for target in self.targets]
        print('\n')

    @staticmethod
    def quantize_audio(data, num_bits=8):
        """Quantize audio
        """
        step_size = 2.0 / 2**(num_bits)
        max_val = 2**(num_bits) - 1
        q_data = np.round((data - (-1.0)) / step_size)
        q_data = np.clip(q_data, 0, max_val)

        return np.uint8(q_data)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):

        rec_num = len(self.data)

        rnd_num = np.random.randint(0, rec_num)
        self.rms[index] = self.rms_val[rnd_num]

        rec_len = len(self.data[rnd_num])

        max_start_idx = rec_len - self.exp_len
        start_idx = np.random.randint(0, max_start_idx)
        end_idx = start_idx + self.exp_len

        inp = self.__reshape_audio(self.data[rnd_num][start_idx:end_idx])
        target = int(self.targets[rnd_num])

        if self.quantize:
            inp /= 256
        if self.transform is not None:
            inp = self.transform(inp)
        return inp, target

    def __reshape_audio(self, audio, row_len=128):

        return torch.transpose(torch.tensor(audio.reshape((-1, row_len))), 1, 0)

    def __gen_datasets(self, exp_len=16384, row_len=128, overlap_ratio=0):

        with warnings.catch_warnings():
            warnings.simplefilter('error')

            # PARAMETERS
            overlap = int(np.ceil(row_len * overlap_ratio))
            num_rows = int(np.ceil(exp_len / (row_len - overlap)))
            data_len = int((num_rows*row_len - (num_rows-1)*overlap))
            print(f'data_len: {data_len}')

            # Cleaning the duplicate labels
            train_list = sorted(os.listdir(self.noise_train_folder))
            test_list = sorted(os.listdir(self.noise_test_folder))
            labels_train = set(sorted({i.split('_')[0] for i in train_list if '_' in i}))
            labels_test = set(sorted({i.split('_')[0] for i in test_list if '_' in i}))
            labels = labels_train | labels_test
            labels_to_remove = set()
            for label in labels:
                other_labels = labels - {label}
                for other_label_name in other_labels:
                    if label in other_label_name:
                        labels_to_remove.add(label)
                        break
            labels = labels - labels_to_remove
            labels = sorted(labels)
            print(f'Labels: {labels}')

            # Folders
            train_test_folders = [self.noise_train_folder, self.noise_test_folder]

            data_in = []
            data_type = []
            data_class = []
            rms_val = []

            for i, label in enumerate(labels):
                for folder in train_test_folders:
                    for record_name in sorted(os.listdir(folder)):
                        if record_name.split('_')[0] in label:
                            record_path = os.path.join(folder, record_name)
                            record, _ = librosa.load(record_path, offset=0, sr=None)

                            if self.quantize:
                                data_in.append(self.quantize_audio(record))
                            else:
                                data_in.append(record)

                            if folder == self.noise_train_folder:
                                data_type.append(0)  # train + val
                            elif folder == self.noise_test_folder:
                                data_type.append(1)  # test

                            data_class.append(i)
                            rms_val.append(np.mean(record**2)**0.5)

            noise_dataset = (data_in, data_class, data_type, rms_val)
        return noise_dataset


def MSnoise_get_datasets(data, load_train=True, load_test=True):
    """
    Load the folded 1D version of MS Scalable Noisy Speech dataset (MS-SNSD)

    The dataset is loaded from the archive file, so the file is required for this version.

    The dataset originally includes 26 different noise types.
    15 of them are chosen classification, others are labeled as the unknown class.
    """
    (data_dir, args) = data

    classes = ['AirConditioner', 'AirportAnnouncements',
               'Babble', 'Bus', 'CafeTeria', 'Car',
               'CopyMachine', 'Field', 'Hallway', 'Kitchen',
               'LivingRoom', 'Metro', 'Munching', 'NeighborSpeaking',
               'Office', 'Park', 'Restaurant', 'ShuttingDoor',
               'Square', 'SqueakyChair', 'Station', 'Traffic',
               'Typing', 'VacuumCleaner', 'WasherDryer', 'Washing', 'TradeShow']

    remove_unknowns = True
    transform = transforms.Compose([
        ai8x.normalize(args=args)
    ])
    quantize = True

    if load_train:
        train_dataset = MSnoise(root=data_dir, classes=classes, d_type='train', dataset_len=11005,
                                remove_unknowns=remove_unknowns, transform=transform,
                                quantize=quantize, download=True)
    else:
        train_dataset = None

    if load_test:
        test_dataset = MSnoise(root=data_dir, classes=classes, d_type='test', dataset_len=11005,
                               remove_unknowns=remove_unknowns, transform=transform,
                               quantize=quantize, download=True)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


def MSnoise_get_unquantized_datasets(data, load_train=True, load_test=True):
    """
    Load the folded 1D and unquantized version of MS Scalable Noisy Speech dataset (MS-SNSD)

    The dataset is loaded from the archive file, so the file is required for this version.

    The dataset originally includes 26 different noise types.
    """
    (data_dir, args) = data

    classes = ['AirConditioner', 'AirportAnnouncements',
               'Babble', 'Bus', 'CafeTeria', 'Car',
               'CopyMachine', 'Field', 'Hallway', 'Kitchen',
               'LivingRoom', 'Metro', 'Munching', 'NeighborSpeaking',
               'Office', 'Park', 'Restaurant', 'ShuttingDoor',
               'Square', 'SqueakyChair', 'Station', 'Traffic',
               'Typing', 'VacuumCleaner', 'WasherDryer', 'Washing', 'TradeShow']

    remove_unknowns = True
    transform = None
    quantize = False

    if load_train:
        train_dataset = MSnoise(root=data_dir, classes=classes, d_type='train', dataset_len=11005,
                                remove_unknowns=remove_unknowns, transform=transform,
                                quantize=quantize, download=True)
    else:
        train_dataset = None

    if load_test:
        test_dataset = MSnoise(root=data_dir, classes=classes, d_type='test', dataset_len=11005,
                               remove_unknowns=remove_unknowns, transform=transform,
                               quantize=quantize, download=True)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'MSnoise',
        'input': (128, 128),
        'output': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        'loader': MSnoise_get_datasets,
    },
    {
        'name': 'MSnoise_unquantized',
        'input': (128, 128),
        'output': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                   16, 17, 18, 19, 20, 21, 22, 23, 24, 25),
        'loader': MSnoise_get_unquantized_datasets,
    },
]
