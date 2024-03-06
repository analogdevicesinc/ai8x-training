###################################################################################################
#
# Copyright (C) 2023-2024 Analog Devices, Inc. All Rights Reserved.
#
# Analog Devices, Inc. Default Copyright Notice:
# https://www.analog.com/en/about-adi/legal-and-risk-oversight/intellectual-property/copyright-notice.html
#
###################################################################################################
#
# Copyright (C) 2019-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
#
# Portions Copyright (c) 2018 Intel Corporation
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
Classes and functions used to create noisy keyword spotting dataset.
"""
import numpy as np
import torch

from datasets import kws20, msnoise


class signalmixer:
    """
    Signal mixer dataloader to create datasets with specified
    length using a noise dataset and a speech dataset and a specified SNR level.

    Args:
    signal_dataset(object): KWS dataset object.
    snr_range(list,int): SNR level range to be used in the snr mixing operation.
    noise_type(string): Noise kind that will be applied to the speech dataset.
    apply_prob(int): Percentage of noise that will be applied to speech dataset.
    noise_dataset(object, optional): MSnoise dataset object.
    """

    def __init__(self, signal_dataset, snr_range, noise_type, apply_prob=1, noise_dataset=None):

        self.data = signal_dataset.data
        self.targets = signal_dataset.targets
        self.signal_dataset = signal_dataset
        self.data_type = signal_dataset.data_type

        if noise_type != ('WhiteNoise' or None):
            self.noise_dataset = noise_dataset

        self.snr_range = snr_range

        if apply_prob > 1:
            print('Noise will be applied to the whole dataset!')
            self.apply_prob = 1
        else:
            self.apply_prob = apply_prob
        self.noise_type = noise_type

    def __getitem__(self, index):
        inp, target = self.signal_dataset[index]
        data_type = self.data_type[index]

        # applying noise to train set using per-snr
        if data_type == 0:
            if np.random.uniform(0, 1) < self.apply_prob:
                if self.noise_type is not None:
                    if not torch.all(inp == 0):
                        inp = self.single_snr_mixer(inp)

        # applying noise to all of the test set
        else:
            if 'WhiteNoise' in self.noise_type:
                if not torch.all(inp == 0):
                    inp = self.white_noise_mixer(inp)
            elif self.noise_type is not None:
                if not torch.all(inp == 0):
                    inp = self.single_snr_mixer(inp)
        return inp.type(torch.FloatTensor), int(target)

    def __len__(self):
        return len(self.data)

    def single_snr_mixer(self, signal):
        '''
        creates mixed signals using an SNR level and the noise dataset
        '''
        idx_noise = np.random.randint(0, len(self.data))
        noise, _ = self.noise_dataset[idx_noise]
        rms_noise = self.noise_dataset.rms[idx_noise]

        snr = self.snr_range
        if snr[0] == snr[-1]:
            snr_val = snr[0]
        else:
            snr_val = np.random.randint(snr[0], snr[-1])
        snr_val = torch.tensor(snr_val)

        rmsclean = torch.sqrt(torch.mean(signal**2))
        scalarclean = 1 / rmsclean
        signal = signal * scalarclean

        scalarnoise = 1 / rms_noise
        noise = noise * scalarnoise

        cleanfactor = 10**(snr_val/20)
        noisyspeech = cleanfactor * signal + noise
        noisyspeech = noisyspeech / (torch.tensor(scalarnoise) + cleanfactor * scalarclean)

        return noisyspeech

    def white_noise_mixer(self, signal):

        '''creates mixed signal dataset using the SNR level and white noise
        '''
        snr = self.snr_range
        if snr[0] == snr[-1]:
            snr_val = snr[0]
        else:
            snr_val = np.random.randint(snr[0], snr[-1])
        snr_val = torch.tensor(snr_val)

        mean = 0
        std = 1
        noise = np.random.normal(mean, std, signal.shape)
        noise = torch.tensor(noise, dtype=torch.float32)

        rmsclean = torch.sqrt(torch.mean(signal**2))
        scalarclean = 1 / rmsclean
        signal = signal * scalarclean

        rmsnoise = torch.sqrt(torch.mean(noise**2))
        scalarnoise = 1 / rmsnoise
        noise = noise * scalarnoise

        cleanfactor = 10**(snr_val/20)
        noisyspeech = cleanfactor * signal + noise
        noisyspeech = noisyspeech / (scalarnoise + cleanfactor * scalarclean)

        return noisyspeech


def signalmixer_get_datasets(data, snr_range, noise_type, desired_probs, apply_prob,
                             load_train=True, load_test=True):
    """
    Returns the KWS dataset mixed with MSnoise dataset.
    """
    (data_dir, _) = data

    kws_train_dataset, kws_test_dataset = kws20.KWS_20_get_datasets(
        data, load_train, load_test)

    if load_train:
        noise_dataset_train = msnoise.MSnoise(root=data_dir, classes=noise_type,
                                              d_type='train', dataset_len=len(kws_train_dataset),
                                              desired_probs=desired_probs,
                                              transform=None, quantize=False, download=False)

        train_dataset = signalmixer(signal_dataset=kws_train_dataset, snr_range=snr_range,
                                    noise_type=noise_type, apply_prob=apply_prob,
                                    noise_dataset=noise_dataset_train)
    else:
        train_dataset = None

    if load_test:
        test_dataset = kws_test_dataset
    else:
        test_dataset = None

    return train_dataset, test_dataset


def signalmixer_all_get_datasets(data, load_train=True, load_test=True):
    """
    Returns the KWS dataset mixed with MSnoise dataset.
    It uses %80 probability for addition of noise.
    Noises are applied between -5 to 10 dB.
    """
    snr_range = range(-5, 10)
    noise_type = ['AirConditioner', 'AirportAnnouncements',
                  'Babble', 'Bus', 'CafeTeria', 'Car',
                  'CopyMachine', 'Field', 'Hallway', 'Kitchen',
                  'LivingRoom', 'Metro', 'Munching', 'NeighborSpeaking',
                  'Office', 'Park', 'Restaurant', 'ShuttingDoor',
                  'Square', 'SqueakyChair', 'Station', 'Traffic',
                  'Typing', 'VacuumCleaner', 'WasherDryer', 'Washing']
    desired_probs = None
    apply_prob = 0.8

    return signalmixer_get_datasets(data=data, snr_range=snr_range, noise_type=noise_type,
                                    desired_probs=desired_probs, apply_prob=apply_prob,
                                    load_train=load_train, load_test=load_test)


def signalmixer_all_100_get_datasets(data, load_train=True, load_test=True):
    """
    Returns the KWS dataset mixed with MSnoise dataset.
    It uses %100 probability for addition of noise.
    Noises are applied between -5 to 10 dB.
    """
    snr_range = range(-5, 10)
    noise_type = ['AirConditioner', 'AirportAnnouncements',
                  'Babble', 'Bus', 'CafeTeria', 'Car',
                  'CopyMachine', 'Field', 'Hallway', 'Kitchen',
                  'LivingRoom', 'Metro', 'Munching', 'NeighborSpeaking',
                  'Office', 'Park', 'Restaurant', 'ShuttingDoor',
                  'Square', 'SqueakyChair', 'Station', 'Traffic',
                  'Typing', 'VacuumCleaner', 'WasherDryer', 'Washing']
    desired_probs = None
    apply_prob = 1

    return signalmixer_get_datasets(data=data, snr_range=snr_range, noise_type=noise_type,
                                    desired_probs=desired_probs, apply_prob=apply_prob,
                                    load_train=load_train, load_test=load_test)


def signalmixer_all_snr_get_datasets(data, load_train=True, load_test=True):
    """
    Returns the KWS dataset mixed with MSnoise dataset.
    It uses %80 probability for addition of noise.
    Noises are applied between 0 to 15 dB.
    """
    snr_range = range(0, 15)
    noise_type = ['AirConditioner', 'AirportAnnouncements',
                  'Babble', 'Bus', 'CafeTeria', 'Car',
                  'CopyMachine', 'Field', 'Hallway', 'Kitchen',
                  'LivingRoom', 'Metro', 'Munching', 'NeighborSpeaking',
                  'Office', 'Park', 'Restaurant', 'ShuttingDoor',
                  'Square', 'SqueakyChair', 'Station', 'Traffic',
                  'Typing', 'VacuumCleaner', 'WasherDryer', 'Washing']
    desired_probs = None
    apply_prob = 0.8

    return signalmixer_get_datasets(data=data, snr_range=snr_range, noise_type=noise_type,
                                    desired_probs=desired_probs, apply_prob=apply_prob,
                                    load_train=load_train, load_test=load_test)


def signalmixer_babble_get_datasets(data, load_train=True, load_test=True):
    """
    Returns the KWS dataset mixed with MSnoise's Babble Noise.
    It uses %80 probability for addition of noise.
    Noises are applied between -5 to 10 dB.
    """
    snr_range = range(-5, 10)
    noise_type = ['Babble']
    desired_probs = None
    apply_prob = 0.8

    return signalmixer_get_datasets(data=data, snr_range=snr_range, noise_type=noise_type,
                                    desired_probs=desired_probs, apply_prob=apply_prob,
                                    load_train=load_train, load_test=load_test)


def signalmixer_vacuum_get_datasets(data, load_train=True, load_test=True):
    """
    Returns the KWS dataset mixed with MSnoise's VacuumCleaner Noise.
    It uses %80 probability for addition of noise.
    Noises are applied between -5 to 10 dB.
    """
    snr_range = range(-5, 10)
    noise_type = ['VacuumCleaner']
    desired_probs = None
    apply_prob = 0.8

    return signalmixer_get_datasets(data=data, snr_range=snr_range, noise_type=noise_type,
                                    desired_probs=desired_probs, apply_prob=apply_prob,
                                    load_train=load_train, load_test=load_test)


def signalmixer_mixed_get_datasets(data, load_train=True, load_test=True):
    """
    Returns the KWS dataset mixed with MSnoise's Babble, VacuumCleaner, Typing & CopyMachine Noise.
    It uses %80 probability for addition of noise.
    Noises are applied between -5 to 10 dB.
    """
    snr_range = range(-5, 10)
    noise_type = ['Babble', 'VacuumCleaner', 'Typing', 'CopyMachine']
    desired_probs = [0.4, 0.2, 0.2, 0.2]
    apply_prob = 0.8

    return signalmixer_get_datasets(data=data, snr_range=snr_range, noise_type=noise_type,
                                    desired_probs=desired_probs, apply_prob=apply_prob,
                                    load_train=load_train, load_test=load_test)


datasets = [
    {
        'name': 'signalmixer',
        'input': (128, 128),
        'output': ('up', 'down', 'left', 'right', 'stop', 'go', 'yes', 'no', 'on', 'off', 'one',
                   'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero',
                   'UNKNOWN'),
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.07),
        'loader': signalmixer_get_datasets,
    },
    {
        'name': 'signalmixer_all',
        'input': (128, 128),
        'output': ('up', 'down', 'left', 'right', 'stop', 'go', 'yes', 'no', 'on', 'off', 'one',
                   'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero',
                   'UNKNOWN'),
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.07),
        'loader': signalmixer_all_get_datasets,
    },
    {
        'name': 'signalmixer_all_100',
        'input': (128, 128),
        'output': ('up', 'down', 'left', 'right', 'stop', 'go', 'yes', 'no', 'on', 'off', 'one',
                   'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero',
                   'UNKNOWN'),
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.07),
        'loader': signalmixer_all_100_get_datasets,
    },
    {
        'name': 'signalmixer_all_snr',
        'input': (128, 128),
        'output': ('up', 'down', 'left', 'right', 'stop', 'go', 'yes', 'no', 'on', 'off', 'one',
                   'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero',
                   'UNKNOWN'),
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.07),
        'loader': signalmixer_all_snr_get_datasets,
    },
    {
        'name': 'signalmixer_babble',
        'input': (128, 128),
        'output': ('up', 'down', 'left', 'right', 'stop', 'go', 'yes', 'no', 'on', 'off', 'one',
                   'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero',
                   'UNKNOWN'),
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.07),
        'loader': signalmixer_babble_get_datasets,
    },
    {
        'name': 'signalmixer_vacuum',
        'input': (128, 128),
        'output': ('up', 'down', 'left', 'right', 'stop', 'go', 'yes', 'no', 'on', 'off', 'one',
                   'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero',
                   'UNKNOWN'),
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.07),
        'loader': signalmixer_vacuum_get_datasets,
    },
    {
        'name': 'signalmixer_mixed',
        'input': (128, 128),
        'output': ('up', 'down', 'left', 'right', 'stop', 'go', 'yes', 'no', 'on', 'off', 'one',
                   'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero',
                   'UNKNOWN'),
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.07),
        'loader': signalmixer_mixed_get_datasets,
    },
]
