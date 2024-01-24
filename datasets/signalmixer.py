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
Classes and functions used to create noisy keyword spotting dataset.
"""
import numpy as np
import torch


class signalmixer:
    """
    Signal mixer dataloader to create datasets with specified
    length using a noise dataset and a speech dataset and a specified SNR level.

    Args:
    signal_dataset(object): KWS dataset object.
    snr(int): SNR level to be created in the mixed dataset.
    noise_kind(string): Noise kind that will be applied to the speech dataset.
    noise_dataset(object, optional): MSnoise dataset object.
    """

    def __init__(self, signal_dataset, snr, noise_kind, noise_dataset=None):

        self.signal_data = signal_dataset.data
        self.signal_targets = signal_dataset.targets

        if noise_kind != 'WhiteNoise':
            self.noise_data = noise_dataset.data
            self.noise_targets = noise_dataset.targets

            # using getitem to reach the noise test data
            self.noise_dataset_float = next(iter(torch.utils.data.DataLoader(
                noise_dataset, batch_size=noise_dataset.dataset_len)))[0]

            self.noise_rms = noise_dataset.rms

        self.snr = snr
        self.noise_kind = noise_kind

        # using getitem to reach the speech test data
        self.test_dataset_float = next(iter(torch.utils.data.DataLoader(
            signal_dataset, batch_size=signal_dataset.data.shape[0])))[0]

        if noise_kind == 'WhiteNoise':
            self.mixed_signal = self.white_noise_mixer()
        else:
            self.mixed_signal = self.snr_mixer()

    def __getitem__(self, index):

        inp = self.mixed_signal[index].type(torch.FloatTensor)
        target = int(self.signal_targets[index])
        return inp, target

    def __len__(self):
        return len(self.mixed_signal)

    def snr_mixer(self):
        ''' creates mixed signal dataset using the SNR level and the noise dataset
        '''
        clean = self.test_dataset_float
        noise = self.noise_dataset_float

        idx = np.random.randint(0, noise.shape[0], clean.shape[0])
        noise = noise[idx]
        rms_noise = self.noise_rms[idx]

        snr = self.snr

        rmsclean = torch.sqrt(torch.mean(clean.reshape(
            clean.shape[0], -1)**2, 1, keepdims=True)).unsqueeze(1)
        scalarclean = 1 / rmsclean
        clean = clean * scalarclean

        scalarnoise = 1 / rms_noise.reshape(-1, 1, 1)
        noise = noise * scalarnoise

        cleanfactor = 10**(snr/20)
        noisyspeech = cleanfactor * clean + noise
        noisyspeech = noisyspeech / (torch.tensor(scalarnoise) + cleanfactor * scalarclean)

        # 16384 --> (noisyspeech[0].shape[0])*(noisyspeech[0].shape[1])
        speech_shape = noisyspeech[0].shape[0]*noisyspeech[0].shape[1]
        max_mixed = torch.max(abs(noisyspeech.reshape(
                        noisyspeech.shape[0], speech_shape)), 1, keepdims=True).values

        noisyspeech = noisyspeech * (1 / max_mixed).unsqueeze(1)
        return noisyspeech

    def white_noise_mixer(self):

        '''creates mixed signal dataset using the SNR level and white noise
        '''
        clean = self.test_dataset_float
        snr = self.snr

        mean = 0
        std = 1
        noise = np.random.normal(mean, std, clean.shape)
        noise = torch.tensor(noise, dtype=torch.float32)

        rmsclean = (torch.mean(clean.reshape(
            clean.shape[0], -1)**2, 1, keepdims=True)**0.5).unsqueeze(1)
        scalarclean = 1 / rmsclean
        clean = clean * scalarclean

        rmsnoise = (torch.mean(noise.reshape(
            noise.shape[0], -1)**2, 1, keepdims=True)**0.5).unsqueeze(1)
        scalarnoise = 1 / rmsnoise
        noise = noise * scalarnoise

        cleanfactor = 10**(snr/20)
        noisyspeech = cleanfactor * clean + noise
        noisyspeech = noisyspeech / (scalarnoise + cleanfactor * scalarclean)

        # scaling to ~[-1,1]
        max_mixed = torch.max(abs(noisyspeech.reshape(
            noisyspeech.shape[0], 16384)), 1, keepdims=True).values
        noisyspeech = noisyspeech * (1 / max_mixed).unsqueeze(1)

        return noisyspeech
