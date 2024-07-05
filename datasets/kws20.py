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
Classes and functions used to create keyword spotting dataset.
"""
import errno
import hashlib
import os
import shutil
import tarfile
import time
import urllib
import warnings
from zipfile import ZipFile

import numpy as np
import torch
import torchaudio
from torch.utils.model_zoo import tqdm  # type: ignore # tqdm exists in model_zoo
from torchvision import transforms

import soundfile as sf

import ai8x
from datasets.msnoise import MSnoise
from datasets.signalmixer import SignalMixer


class KWS:
    """
    `SpeechCom v0.02 <http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz>`
    Dataset, 1D folded.

    Args:
    root (string): Root directory of dataset where ``KWS/processed/dataset.pt`` exist.
    classes(array): List of keywords to be used.
    d_type(string): Option for the created dataset. ``train`` or ``test``.
    transform (callable, optional): A function/transform that takes in a signal between [0, 1]
        and returns a transformed version, suitable for ai8x training / evaluation.
    quantization_scheme (dict, optional): Dictionary containing quantization scheme parameters.
        If not provided, default values are used.
    augmentation (dict, optional): Dictionary containing augmentation parameters.
    download (bool, optional): If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.
    save_unquantized (bool, optional): If true, folded but unquantized data is saved.
    filter_libri (bool, optional): If true, librispeech class will be eliminated from the dataset.
    benchmark (bool, optional): If true, benchmark dataset will be loaded.

    """

    url_speechcommand = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
    url_test = 'http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz'
    url_librispeech = 'http://us.openslr.org/resources/12/dev-clean.tar.gz'
    fs = 16000

    class_dict = {'_silence_': 0, 'backward': 1, 'bed': 2, 'bird': 3, 'cat': 4, 'dog': 5,
                  'down': 6, 'eight': 7, 'five': 8, 'follow': 9, 'forward': 10, 'four': 11,
                  'go': 12, 'happy': 13, 'house': 14, 'learn': 15, 'left': 16, 'librispeech': 17,
                  'marvin': 18, 'nine': 19, 'no': 20, 'off': 21, 'on': 22, 'one': 23, 'right': 24,
                  'seven': 25, 'sheila': 26, 'six': 27, 'stop': 28, 'three': 29, 'tree': 30,
                  'two': 31, 'up': 32, 'visual': 33, 'wow': 34, 'yes': 35, 'zero': 36}

    dataset_dict = {
        'KWS': ('up', 'down', 'left', 'right', 'stop', 'go', '_unknown_'),
        'KWS_20': ('up', 'down', 'left', 'right', 'stop', 'go', 'yes', 'no', 'on', 'off', 'one',
                   'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero',
                   '_unknown_'),
        'KWS_35': ('backward', 'bed', 'bird', 'cat', 'dog', 'down',
                   'eight', 'five', 'follow', 'forward', 'four', 'go',
                   'happy', 'house', 'learn', 'left', 'marvin', 'nine',
                   'no', 'off', 'on', 'one', 'right', 'seven',
                   'sheila', 'six', 'stop', 'three', 'tree', 'two',
                   'up', 'visual', 'wow', 'yes', 'zero', '_unknown_'),
        'KWS_12_benchmark': ('down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes',
                             '_silence_', '_unknown_')
    }

    benchmark_keywords = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop',
                          'up', 'yes', '_silence_']

    # define constants for data types (train, test, validation, benchmark)
    TRAIN = np.uint(0)
    TEST = np.uint(1)
    VALIDATION = np.uint(2)
    BENCHMARK = np.uint(3)

    def __init__(self, root, classes, d_type, t_type, transform=None, quantization_scheme=None,
                 augmentation=None, download=False, save_unquantized=False, filter_libri=False,
                 benchmark=False):

        self.root = root
        self.classes = classes
        self.d_type = d_type
        self.t_type = t_type
        self.transform = transform
        self.save_unquantized = save_unquantized
        self.filter_libri = filter_libri
        self.benchmark = benchmark

        self.__parse_quantization(quantization_scheme)
        self.__parse_augmentation(augmentation)

        if not self.save_unquantized:
            self.data_file = 'dataset.pt'
        else:
            self.data_file = 'unquantized.pt'

        if download:
            self.__download()

        self.data, self.targets, self.data_type, self.shift_limits = \
            torch.load(os.path.join(self.processed_folder, self.data_file))

        print(f'\nProcessing {self.d_type}...')
        self.__filter_dtype()

        if '_silence_' not in self.classes:
            self.__filter_silence()

        if self.filter_libri:
            self.__filter_librispeech()

        self.__filter_classes()

    @property
    def raw_folder(self):
        """Folder for the raw data.
        """
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def raw_test_folder(self):
        """Folder for the raw data.
        """
        return os.path.join(self.root, self.__class__.__name__, 'raw_test')

    @property
    def librispeech_folder(self):
        """Folder for the librispeech data.
        """
        return os.path.join(self.root, self.__class__.__name__, 'librispeech')

    @property
    def noise_folder(self):
        """Folder for the different noise data.
        """
        return os.path.join(self.root, self.__class__.__name__, 'noise')

    @property
    def processed_folder(self):
        """Folder for the processed data.
        """
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def silence_folder(self):
        """Folder for the silence data.
        """
        return os.path.join(self.raw_folder,  '_silence_')

    def __parse_quantization(self, quantization_scheme):
        if quantization_scheme:
            self.quantization = quantization_scheme
            if 'bits' not in self.quantization:
                self.quantization['bits'] = 8
            if self.quantization['bits'] == 0:
                self.save_unquantized = True
            if 'compand' not in self.quantization:
                self.quantization['compand'] = False
            if 'mu' not in self.quantization:
                self.quantization['mu'] = 255  # Default, ignored when 'compand' is False
        else:
            print('Undefined quantization schema! ',
                  'Number of bits set to 8.')
            self.quantization = {'bits': 8, 'compand': False, 'mu': 255}

    def __parse_augmentation(self, augmentation):
        self.augmentation = augmentation
        if augmentation:
            if 'aug_num' not in augmentation:
                print('No key `aug_num` in input augmentation dictionary! ',
                      'Using 0.')
                self.augmentation['aug_num'] = 0
            elif self.augmentation['aug_num'] != 0:
                if 'snr' not in augmentation:
                    print('No key `snr` in input augmentation dictionary! ',
                          'Using defaults: [Min: -5.0, Max: 20.0]')
                    self.augmentation['snr'] = {'min': -5.0, 'max': 20.0}
                if 'shift' not in augmentation:
                    print('No key `shift` in input augmentation dictionary! '
                          'Using defaults: [Min:-0.1, Max: 0.1]')
                    self.augmentation['shift'] = {'min': -0.1, 'max': 0.1}

    def __download(self):

        if self.__check_exists():
            return

        self.__makedir_exist_ok(self.raw_folder)
        self.__makedir_exist_ok(self.processed_folder)

        # download Google Speech Commands dataset
        filename = self.url_speechcommand.rpartition('/')[2]
        self.__download_and_extract_archive(self.url_speechcommand,
                                            download_root=self.raw_folder,
                                            filename=filename)

        # sample long segments of background noise (total 560*6 = 3360 samples)
        self.__sample_wav(os.path.join(self.raw_folder, '_background_noise_'),
                          self.silence_folder, 560)

        # download Google Speech Commands official test set for 10 keywords + _silence_ + _unknown_
        filename = self.url_test.rpartition('/')[2]
        self.__download_and_extract_archive(self.url_test,
                                            download_root=self.raw_test_folder,
                                            filename=filename)

        # copy test silence files to raw folder, for ease of processing in gen_datasets
        shutil.copytree(os.path.join(self.raw_test_folder, '_silence_'),
                        os.path.join(self.raw_folder, '_silence_'), dirs_exist_ok=True)

        print('Test for _silence_ class successfully copied under raw/_silence_ folder.')

        # download LibriSpeech dev-clean dataset
        filename = self.url_librispeech.rpartition('/')[2]
        self.__download_and_extract_archive(self.url_librispeech,
                                            download_root=self.librispeech_folder,
                                            filename=filename)

        # convert the LibriSpeech audio files to 1-sec 16KHz .wav, store under raw/librispeech
        self.__resample_convert_wav(folder_in=self.librispeech_folder,
                                    folder_out=os.path.join(self.raw_folder, 'librispeech'))

        self.__gen_datasets()

    def __sample_wav(self, folder_in, folder_out, sample_num, sr=16000, ext='.wav', exp_len=16384):

        # create output folder
        self.__makedir_exist_ok(folder_out)

        for (dirpath, _, filenames) in os.walk(folder_in):
            for filename in sorted(filenames):
                if filename.endswith(ext):
                    fname = os.path.join(dirpath, filename)
                    data, sr_original = sf.read(fname, dtype='float32')
                    assert sr_original == sr
                    max_start_pt = len(data) - exp_len

                    for i in range(sample_num):
                        start_pt = np.random.randint(0, max_start_pt)
                        audio = data[start_pt:start_pt+exp_len]

                        outfile = os.path.join(folder_out, filename[:-len(ext)] + '_' +
                                               str(f"{i}") + '.wav')
                        sf.write(outfile, audio, sr)
        print('File conversion completed.')

    def __check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.data_file))

    def __makedir_exist_ok(self, dirpath):
        try:
            os.makedirs(dirpath)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

    def __gen_bar_updater(self):
        pbar = tqdm(total=None)

        def bar_update(count, block_size, total_size):
            if pbar.total is None and total_size:
                pbar.total = total_size
            progress_bytes = count * block_size
            pbar.update(progress_bytes - pbar.n)

        return bar_update

    def __download_url(self, url, root, filename=None, md5=None):
        root = os.path.expanduser(root)
        if not filename:
            filename = os.path.basename(url)
        fpath = os.path.join(root, filename)

        self.__makedir_exist_ok(root)

        # downloads file
        if self.__check_integrity(fpath, md5):
            print('Using downloaded and verified file: ' + fpath)
        else:
            try:
                print('Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath, reporthook=self.__gen_bar_updater())
            except (urllib.error.URLError, IOError) as e:
                if url[:5] == 'https':
                    url = url.replace('https:', 'http:')
                    print('Failed download. Trying https -> http instead.'
                          ' Downloading ' + url + ' to ' + fpath)
                    urllib.request.urlretrieve(url, fpath, reporthook=self.__gen_bar_updater())
                else:
                    raise e

    def __calculate_md5(self, fpath, chunk_size=1024 * 1024):
        md5 = hashlib.md5()
        with open(fpath, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                md5.update(chunk)
        return md5.hexdigest()

    def __check_md5(self, fpath, md5, **kwargs):
        return md5 == self.__calculate_md5(fpath, **kwargs)

    def __check_integrity(self, fpath, md5=None):
        if not os.path.isfile(fpath):
            return False
        if md5 is None:
            return True
        return self.__check_md5(fpath, md5)

    def __extract_archive(self, from_path,
                          to_path=None, remove_finished=False):
        if to_path is None:
            to_path = os.path.dirname(from_path)

        if from_path.endswith('.tar.gz'):
            with tarfile.open(from_path, 'r:gz') as tar:
                tar.extractall(path=to_path, filter='data')
        elif from_path.endswith('.zip'):
            with ZipFile(from_path) as archive:
                archive.extractall(to_path)
        else:
            raise ValueError(f"Extraction of {from_path} not supported")

        if remove_finished:
            os.remove(from_path)

    def __download_and_extract_archive(self, url, download_root, extract_root=None, filename=None,
                                       md5=None, remove_finished=False):
        download_root = os.path.expanduser(download_root)
        if extract_root is None:
            extract_root = download_root
        if not filename:
            filename = os.path.basename(url)

        self.__download_url(url, download_root, filename, md5)

        archive = os.path.join(download_root, filename)
        print(f"Extracting {archive} to {extract_root}")
        self.__extract_archive(archive, extract_root, remove_finished)

    def __resample_convert_wav(self, folder_in, folder_out, sr=16000, ext='.flac'):
        # create output folder
        self.__makedir_exist_ok(folder_out)

        # find total number of files to convert
        total_count = 0
        for (dirpath, _, filenames) in os.walk(folder_in):
            for filename in sorted(filenames):
                if filename.endswith(ext):
                    total_count += 1
        print(f"Total number of speech files to convert to 1-sec .wav: {total_count}")
        converted_count = 0
        # segment each audio file to 1-sec frames and save
        for (dirpath, _, filenames) in os.walk(folder_in):
            for filename in sorted(filenames):

                i = 0
                if filename.endswith(ext):
                    fname = os.path.join(dirpath, filename)
                    data, sr_original = sf.read(fname, dtype='float32')
                    assert sr_original == sr

                    # normalize data
                    mx = np.amax(abs(data))
                    data = data / mx

                    chunk_start = 0
                    frame_count = 0

                    # The beginning of an utterance is detected when the average
                    # of absolute values of 128-sample chunks is above a threshold.
                    # Then, a segment is formed from 30*128 samples before the beginning
                    # of the utterance to 98*128 samples after that.
                    # This 1 second (16384 samples) audio segment is converted to .wav
                    # and saved in librispeech folder together with other keywords to
                    # be used as the unknown class.

                    precursor_len = 30 * 128
                    postcursor_len = 98 * 128
                    utterance_threshold = 30

                    while True:
                        if chunk_start + postcursor_len > len(data):
                            break

                        chunk = data[chunk_start: chunk_start + 128]
                        # scaled average over 128 samples
                        avg = 1000 * np.average(abs(chunk))
                        i += 128

                        if avg > utterance_threshold and chunk_start >= precursor_len:
                            print(f"\r Converting {converted_count + 1}/{total_count} "
                                  f"to {frame_count + 1} segments", end=" ")
                            frame = data[chunk_start - precursor_len:chunk_start + postcursor_len]

                            outfile = os.path.join(folder_out, filename[:-5] + '_' +
                                                   str(f"{frame_count}") + '.wav')
                            sf.write(outfile, frame, sr)

                            chunk_start += postcursor_len
                            frame_count += 1
                        else:
                            chunk_start += 128
                    converted_count += 1
                else:
                    pass
        print(f'\rFile conversion completed: {converted_count} files ')

    def __filter_dtype(self):
        if self.d_type == 'train':
            idx_to_select = (self.data_type == self.TRAIN)[:, -1]
        elif self.d_type == 'test':
            if self.benchmark:
                idx_to_select = (self.data_type == self.BENCHMARK)[:, -1]
            else:
                idx_bm = (self.data_type == self.BENCHMARK)[:, -1]
                idx_test = (self.data_type == self.TEST)[:, -1]
                idx_to_select = torch.logical_or(idx_bm, idx_test)
        else:
            print(f'Unknown data type: {self.d_type}')
            return

        set_size = idx_to_select.sum()
        print(f'{self.d_type} set: {set_size} elements')
        # take a copy of the original data and targets temporarily for validation set
        self.data_original = self.data.clone()
        self.targets_original = self.targets.clone()
        self.data_type_original = self.data_type.clone()
        self.shift_limits_original = self.shift_limits.clone()

        self.data = self.data[idx_to_select, :]
        self.targets = self.targets[idx_to_select, :]
        if self.d_type == 'test':
            self.data_type[idx_to_select, :] = self.TEST

        self.data_type = self.data_type[idx_to_select, :]
        self.shift_limits = self.shift_limits[idx_to_select, :]

        # append validation set to the training set if validation examples are explicitly included
        if self.d_type == 'train':
            idx_to_select = (self.data_type_original == self.VALIDATION)[:, -1]
            if idx_to_select.sum() > 0:  # if validation examples exist
                self.data = torch.cat((self.data, self.data_original[idx_to_select, :]), dim=0)
                self.targets = \
                    torch.cat((self.targets, self.targets_original[idx_to_select, :]), dim=0)
                self.data_type = \
                    torch.cat((self.data_type, self.data_type_original[idx_to_select, :]), dim=0)
                self.shift_limits = \
                    torch.cat((self.shift_limits, self.shift_limits_original[idx_to_select, :]),
                              dim=0)
                # indicate the list of validation indices to be used by distiller's dataloader
                self.valid_indices = range(set_size, set_size + idx_to_select.sum())
                print(f'validation set: {idx_to_select.sum()} elements')

        del self.data_original
        del self.targets_original
        del self.data_type_original
        del self.shift_limits_original

    def __filter_classes(self):
        initial_new_class_label = len(self.class_dict)
        new_class_label = initial_new_class_label
        for c in self.classes:
            if c not in self.class_dict:
                if c == '_unknown_':
                    continue
                raise ValueError(f'Class {c} not found in data')
            num_elems = (self.targets == self.class_dict[c]).cpu().sum()
            print(f'Class {c} (# {self.class_dict[c]}): {num_elems} elements')
            self.targets[(self.targets == self.class_dict[c])] = new_class_label
            new_class_label += 1

        num_elems = (self.targets < initial_new_class_label).cpu().sum()
        print(f'Class _unknown_: {num_elems} elements')
        self.targets[(self.targets < initial_new_class_label)] = new_class_label
        self.targets -= initial_new_class_label

    def __filter_librispeech(self):

        print('Filtering out librispeech elements...')
        idx_for_librispeech = [idx for idx, val in enumerate(self.targets)
                               if val != self.class_dict['librispeech']]

        self.data = torch.index_select(self.data, 0, torch.tensor(idx_for_librispeech))
        self.targets = torch.index_select(self.targets, 0, torch.tensor(idx_for_librispeech))
        self.data_type = torch.index_select(self.data_type, 0, torch.tensor(idx_for_librispeech))
        self.shift_limits = torch.index_select(self.shift_limits, 0,
                                               torch.tensor(idx_for_librispeech))

        if self.d_type == 'train':
            set_size = sum((self.data_type == self.TRAIN)[:, -1])
        elif self.d_type == 'test':
            set_size = sum((self.data_type == self.TEST)[:, -1])
        print(f'Remaining {self.d_type} set: {set_size} elements')

        if self.d_type == 'train':
            train_size = sum((self.data_type == self.TRAIN)[:, -1])
            set_size = sum((self.data_type == self.VALIDATION)[:, -1])
            # indicate the list of validation indices to be used by distiller's dataloader
            self.valid_indices = range(train_size, train_size + set_size)
            print(f'Remaining validation set: {set_size} elements')

    def __filter_silence(self):

        print('Filtering out _silence_ elements...')
        idx_for_silence = [idx for idx, val in enumerate(self.targets)
                           if val != self.class_dict['_silence_']]

        self.data = torch.index_select(self.data, 0, torch.tensor(idx_for_silence))
        self.targets = torch.index_select(self.targets, 0, torch.tensor(idx_for_silence))
        self.data_type = torch.index_select(self.data_type, 0, torch.tensor(idx_for_silence))
        self.shift_limits = torch.index_select(self.shift_limits, 0, torch.tensor(idx_for_silence))

        if self.d_type == 'train':
            set_size = sum((self.data_type == self.TRAIN)[:, -1])
        elif self.d_type == 'test':
            set_size = sum((self.data_type == self.TEST)[:, -1])
        print(f'Remaining {self.d_type} set: {set_size} elements')

        if self.d_type == 'train':
            train_size = sum((self.data_type == self.TRAIN)[:, -1])
            set_size = sum((self.data_type == self.VALIDATION)[:, -1])
            # indicate the list of validation indices to be used by distiller's dataloader
            self.valid_indices = range(train_size, train_size + set_size)
            print(f'Remaining validation set: {set_size} elements')

    def __len__(self):
        return len(self.data)

    def __reshape_audio(self, audio, row_len=128):
        # add overlap if necessary later on
        return torch.transpose(audio.reshape((-1, row_len)), 1, 0)

    def shift_and_noise_augment(self, audio, shift_limits):
        """Augments audio by adding random shift and noise.
        """
        random_shift_sample = np.random.randint(shift_limits[0], shift_limits[1])
        aug_audio = self.shift(audio, random_shift_sample)

        if 'snr' in self.augmentation:
            random_snr_coeff = int(np.random.uniform(self.augmentation['snr']['min'],
                                                     self.augmentation['snr']['max']))
            random_snr_coeff = 10 ** (random_snr_coeff / 10)
            if self.quantization['bits'] == 0:
                aug_audio = self.add_white_noise(aug_audio, random_snr_coeff)
            else:
                aug_audio = self.add_quantized_white_noise(aug_audio, random_snr_coeff)

        return aug_audio

    def __getitem__(self, index):
        inp, target = self.data[index], int(self.targets[index])
        data_type, shift_limits = self.data_type[index], self.shift_limits[index]

        # apply dynamic shift and noise augmentation to training examples
        if data_type == self.TRAIN:
            inp = self.shift_and_noise_augment(inp, shift_limits)

        # reshape to 2D
        inp = self.__reshape_audio(inp)
        inp = inp.type(torch.FloatTensor)

        if not self.save_unquantized:
            inp /= 256
        if self.transform is not None:
            inp = self.transform(inp)

        return inp, target

    @staticmethod
    def add_white_noise(audio, random_snr_coeff):
        """Adds zero mean Gaussian noise to signal with specified SNR value.
        """
        signal_var = torch.var(audio)
        noise_var_coeff = signal_var / random_snr_coeff
        noise = np.random.normal(0, torch.sqrt(noise_var_coeff), len(audio))
        return audio + torch.Tensor(noise)

    @staticmethod
    def add_quantized_white_noise(audio, random_snr_coeff):
        """Adds zero mean Gaussian noise to signal with specified SNR value.
        """
        signal_var = torch.var(audio.type(torch.float))
        noise_var_coeff = signal_var / random_snr_coeff
        noise = np.random.normal(0, torch.sqrt(noise_var_coeff), len(audio))
        noise = torch.Tensor(noise).type(torch.int16)
        return (audio + noise).clip(0, 255).type(torch.uint8)

    @staticmethod
    def shift(audio, shift_sample):
        """Shifts audio.
        """
        return torch.roll(audio, shift_sample)

    @staticmethod
    def compand(data, mu=255):
        """Compand the signal level to warp from Laplacian distribution to uniform distribution"""
        data = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(1 + mu)
        return data

    @staticmethod
    def expand(data, mu=255):
        """Undo the companding"""
        data = np.sign(data) * (1 / mu) * (np.power((1 + mu), np.abs(data)) - 1)
        return data

    @staticmethod
    def quantize_audio(data, num_bits=8, compand=False, mu=255):
        """Quantize audio
        """
        if compand:
            data = KWS.compand(data, mu)

        step_size = 2.0 / 2 ** (num_bits)
        max_val = 2 ** (num_bits) - 1
        q_data = np.round((data - (-1.0)) / step_size)
        q_data = np.clip(q_data, 0, max_val)

        if compand:
            data_ex = (q_data - 2 ** (num_bits - 1)) / 2 ** (num_bits - 1)
            data_ex = KWS.expand(data_ex)
            q_data = np.round((data_ex - (-1.0)) / step_size)
            q_data = np.clip(q_data, 0, max_val)
        return np.uint8(q_data)

    def get_audio_endpoints(self, audio, fs):
        """Future: May implement a method to detect the beginning & end of voice activity in audio.
        Currently, it returns end points compatible with augmentation['shift'] values
        """
        if self.augmentation:
            return int(-self.augmentation['shift']['min'] * fs), \
                int(len(audio) - self.augmentation['shift']['max'] * fs)

        return (0, int(len(audio)) - 1)

    def speed_augment(self, audio, fs, sample_no=0):
        """Augments audio by randomly changing the speed of the audio.
        The generated coefficient follows 0.9, 1.1, 0.95, 1.05... pattern
        """
        speed_multiplier = 1.0 + 0.2 * (sample_no % 2 - 0.5) / (1 + sample_no // 2)

        sox_effects = [["speed", str(speed_multiplier)], ["rate", str(fs)]]
        aug_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            torch.unsqueeze(torch.from_numpy(audio).float(), dim=0), fs, sox_effects)
        aug_audio = aug_audio.numpy().squeeze()

        return aug_audio, speed_multiplier

    def speed_augment_multiple(self, audio, fs, exp_len, n_augment):
        """Calls `speed_augment` function for n_augment times for given audio data.
        Finally the original audio is added to have (n_augment+1) audio data.
        """
        aug_audio = [None] * (n_augment + 1)
        aug_speed = np.ones((n_augment + 1,))
        shift_limits = np.zeros((n_augment + 1, 2))
        voice_begin_idx, voice_end_idx = self.get_audio_endpoints(audio, fs)
        aug_audio[0] = audio
        for i in range(n_augment):
            aug_audio[i+1], aug_speed[i+1] = self.speed_augment(audio, fs, sample_no=i)
        for i in range(n_augment + 1):
            if len(aug_audio[i]) < exp_len:
                aug_audio[i] = np.pad(aug_audio[i], (0, exp_len - len(aug_audio[i])), 'constant')
            aug_begin_idx = voice_begin_idx * aug_speed[i]
            aug_end_idx = voice_end_idx * aug_speed[i]
            if aug_end_idx - aug_begin_idx <= exp_len:
                # voice activity duration is shorter than the expected length
                segment_begin = max(aug_end_idx, exp_len) - exp_len
                segment_end = max(aug_end_idx, exp_len)
                aug_audio[i] = aug_audio[i][segment_begin:segment_end]
                shift_limits[i, 0] = -aug_begin_idx + (max(aug_end_idx, exp_len) - exp_len)
                shift_limits[i, 1] = max(aug_end_idx, exp_len) - aug_end_idx
            else:
                # voice activity duraction is longer than the expected length
                midpoint = (aug_begin_idx + aug_end_idx) // 2
                aug_audio[i] = aug_audio[i][midpoint - exp_len // 2: midpoint + exp_len // 2]
                shift_limits[i, :] = [0, 0]
        return aug_audio, shift_limits

    def __gen_datasets(self, exp_len=16384):
        print('Generating dataset from raw data samples for the first time.')
        print('This process may take a few minutes.')
        with warnings.catch_warnings():
            warnings.simplefilter('error')

            lst = sorted(os.listdir(self.raw_folder))
            labels = [d for d in lst if os.path.isdir(os.path.join(self.raw_folder, d))
                      and d[0].isalpha() or d == '_silence_']

            # show the size of dataset for each keyword
            print('------------- Label Size ---------------')
            for i, label in enumerate(labels):
                record_list = os.listdir(os.path.join(self.raw_folder, label))
                print(f'{label:8s}:  \t{len(record_list)}')
            print('------------------------------------------')

            # read testing_list.txt & validation_list.txt into sets for fast access
            with open(os.path.join(self.raw_folder, 'testing_list.txt'), encoding="utf-8") as f:
                test_set = set(f.read().splitlines())
            test_silence = [os.path.join('_silence_', rec) for rec in os.listdir(
                os.path.join(self.raw_test_folder, '_silence_'))]
            test_set.update(test_silence)

            with open(os.path.join(self.raw_folder, 'validation_list.txt'), encoding="utf-8") as f:
                validation_set = set(f.read().splitlines())

            # sample 1 out of every 9 generated silence files for the validation set
            silence_files = [os.path.join('_silence_', s) for s in os.listdir(self.silence_folder)
                             if not s[0].isdigit()]  # files starting w/ numbers: used for testing
            validation_set.update(silence_files[::9])

            train_count = 0
            test_count = 0
            valid_count = 0

            for i, label in enumerate(labels):
                print(f'Processing the label: {label}. {i + 1} of {len(labels)}')
                record_list = sorted(os.listdir(os.path.join(self.raw_folder, label)))
                record_len = len(record_list)

                # get the number testing samples for the class
                test_count_class = 0
                for r, record_name in enumerate(record_list):
                    local_filename = os.path.join(label, record_name)
                    if local_filename in test_set:
                        test_count_class += 1

                # no augmentation for test set, subtract them accordingly
                number_of_total_samples = record_len * (self.augmentation['aug_num'] + 1) - \
                    test_count_class * self.augmentation['aug_num']

                if not self.save_unquantized:
                    data_in = np.empty((number_of_total_samples, exp_len), dtype=np.uint8)
                else:
                    data_in = np.empty((number_of_total_samples, exp_len), dtype=np.float32)

                data_type = np.empty((number_of_total_samples, 1), dtype=np.uint8)
                data_shift_limits = np.empty((number_of_total_samples, 2), dtype=np.int16)
                data_class = np.full((number_of_total_samples, 1), i, dtype=np.uint8)

                time_s = time.time()

                sample_index = 0
                for r, record_name in enumerate(record_list):

                    record_name = os.path.join(label, record_name)

                    if r % 1000 == 0:
                        print(f'\t{r + 1} of {record_len}')

                    if label in self.benchmark_keywords:
                        if label == '_silence_':
                            raw_test_list = os.listdir(os.path.join(
                                self.raw_test_folder, '_silence_'))
                        else:
                            raw_test_list = os.listdir(os.path.join(self.raw_test_folder, label))
                        raw_test_list = [os.path.join(label, rec) for rec in raw_test_list]
                    else:
                        raw_test_list = os.listdir(os.path.join(self.raw_test_folder, '_unknown_'))
                        # example record name: "backward_a6f2fd71_nohash_3.wav.wav"
                        # note: the double "wav" extension is due to errors in the original dataset
                        raw_test_list = [os.path.join(label, rec[-25:-4]) for rec in raw_test_list
                                         if label in rec]

                    if record_name in raw_test_list:
                        d_typ = self.BENCHMARK  # benchmark test
                        test_count += 1
                    elif record_name in test_set:
                        d_typ = self.TEST
                        test_count += 1
                    elif record_name in validation_set:
                        d_typ = self.VALIDATION
                        valid_count += 1
                    else:
                        d_typ = self.TRAIN
                        train_count += 1

                    record_pth = os.path.join(self.raw_folder, record_name)
                    record, fs = sf.read(record_pth, dtype='float32')

                    # training and validation examples get speed augmentation
                    if d_typ not in (self.TEST, self.BENCHMARK):
                        no_augmentations = self.augmentation['aug_num']
                    else:  # test examples don't get speed augmentation
                        no_augmentations = 0

                    # apply speed augmentations and calculate shift limits
                    audio_seq_list, shift_limits = \
                        self.speed_augment_multiple(record, fs, exp_len, no_augmentations)

                    for local_id, audio_seq in enumerate(audio_seq_list):
                        if not self.save_unquantized:
                            data_in[sample_index] = \
                                KWS.quantize_audio(audio_seq,
                                                   num_bits=self.quantization['bits'],
                                                   compand=self.quantization['compand'],
                                                   mu=self.quantization['mu'])
                        else:
                            data_in[sample_index] = audio_seq
                        data_shift_limits[sample_index] = shift_limits[local_id]
                        data_type[sample_index] = d_typ
                        sample_index += 1

                dur = time.time() - time_s
                print(f'Finished in {dur:.3f} seconds.')
                print(data_in.shape)
                time_s = time.time()
                if i == 0:
                    data_in_all = data_in.copy()
                    data_class_all = data_class.copy()
                    data_type_all = data_type.copy()
                    data_shift_limits_all = data_shift_limits.copy()
                else:
                    data_in_all = np.concatenate((data_in_all, data_in), axis=0)
                    data_class_all = np.concatenate((data_class_all, data_class), axis=0)
                    data_type_all = np.concatenate((data_type_all, data_type), axis=0)
                    data_shift_limits_all = \
                        np.concatenate((data_shift_limits_all, data_shift_limits), axis=0)
                dur = time.time() - time_s
                print(f'Data concatenation finished in {dur:.3f} seconds.')

            data_in_all = torch.from_numpy(data_in_all)
            data_class_all = torch.from_numpy(data_class_all)
            data_type_all = torch.from_numpy(data_type_all)
            data_shift_limits_all = torch.from_numpy(data_shift_limits_all)

            # apply static shift & noise augmentation for validation examples
            for sample_index in range(data_in_all.shape[0]):
                if data_type_all[sample_index] == self.VALIDATION:
                    data_in_all[sample_index] = \
                        self.shift_and_noise_augment(data_in_all[sample_index],
                                                     data_shift_limits_all[sample_index])

            raw_dataset = (data_in_all, data_class_all, data_type_all, data_shift_limits_all)
            torch.save(raw_dataset, os.path.join(self.processed_folder, self.data_file))

        print('Dataset created.')
        print(f'Training: {train_count}, Validation: {valid_count}, Test: {test_count}')


def KWS_get_datasets(data, load_train=True, load_test=True, dataset_name='KWS', num_classes=6,
                     quantized=True, filter_libri=False, benchmark=False):
    """
    Load the folded 1D version of SpeechCom dataset

    The dataset is loaded from the archive file, so the file is required for this version.

    The dataset originally includes 35 keywords. A dataset is formed with num_classes + 1 classes
    which includes num_classes of the original keywords and the rest of the dataset is used to
    form the last class, i.e class of the unknowns.
    To further improve the detection of unknown words, the librispeech dataset is also downloaded
    and converted to 1 second segments to be used as unknowns as well.
    The dataset is split into training+validation and test sets. 90:10 training+validation:test
    split is used by default.

    Data is augmented to 3x duplicate data by random stretch/shift and randomly adding noise where
    the stretching coefficient, shift amount and noise SNR level are randomly selected between
    0.8 and 1.3, -0.1 and 0.1, -5 and 20, respectively.
    """
    (data_dir, args) = data

    if quantized:
        transform = transforms.Compose([
            ai8x.normalize(args=args)
        ])
    else:
        transform = None

    classes = KWS.dataset_dict[dataset_name]

    if num_classes+1 != len(classes):
        raise ValueError(f'num_classes {num_classes}does not match with classes')

    if quantized:
        augmentation = {'aug_num': 2, 'shift': {'min': -0.1, 'max': 0.1},
                        'snr': {'min': -5.0, 'max': 20.}}
        quantization_scheme = {'compand': False, 'mu': 10}
    else:
        # default: no speed augmentation for unquantized due to memory usage considerations
        augmentation = {'aug_num': 0, 'shift': {'min': -0.1, 'max': 0.1},
                        'snr': {'min': -5.0, 'max': 20.}}
        quantization_scheme = {'bits': 0}

    if load_train:
        train_dataset = KWS(root=data_dir, classes=classes, d_type='train',
                            transform=transform, t_type='keyword',
                            quantization_scheme=quantization_scheme,
                            augmentation=augmentation, download=True,
                            filter_libri=filter_libri,
                            benchmark=benchmark)
    else:
        train_dataset = None

    if load_test:
        test_dataset = KWS(root=data_dir, classes=classes, d_type='test',
                           transform=transform, t_type='keyword',
                           quantization_scheme=quantization_scheme,
                           augmentation=augmentation, download=True,
                           filter_libri=filter_libri,
                           benchmark=benchmark)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


def KWS_20_get_datasets(data, load_train=True, load_test=True):
    """
    Load the folded 1D version of SpeechCom dataset for 20 classes

    The dataset is loaded from the archive file, so the file is required for this version.

    The dataset originally includes 35 keywords. A dataset is formed with 21 classes which includes
    20 of the original keywords and the rest of the dataset is used to form the last class,
    i.e class of the unknowns.
    To further improve the detection of unknown words, the librispeech dataset is also downloaded
    and converted to 1 second segments to be used as unknowns as well.
    The dataset is split into training+validation and test sets. 90:10 training+validation:test
    split is used by default.

    Data is augmented to 3x duplicate data by random stretch/shift and randomly adding noise where
    the stretching coefficient, shift amount and noise variance are randomly selected between
    0.8 and 1.3, -0.1 and 0.1, 0 and 1, respectively.
    """
    return KWS_get_datasets(data, load_train, load_test, dataset_name='KWS_20', num_classes=20)


def KWS_35_get_datasets(data, load_train=True, load_test=True, benchmark=False):
    """
    Load the folded 1D version of SpeechCom dataset for 35 classes

    The dataset is loaded from the archive file, so the file is required for this version.

    The dataset includes 35 keywords. The librispeech dataset is also downloaded
    and converted to 1 second segments to be used as unknowns.
    The dataset is split into training+validation and test sets. 90:10 training+validation:test
    split is used by default.

    Data is augmented to 3x duplicate data by random stretch/shift and randomly adding noise where
    the stretching coefficient, shift amount and noise variance are randomly selected between
    0.8 and 1.3, -0.1 and 0.1, 0 and 1, respectively.
    """
    return KWS_get_datasets(data, load_train, load_test,  dataset_name='KWS_35',
                            num_classes=35, benchmark=benchmark)


def KWS_35_get_unquantized_datasets(data, load_train=True, load_test=True):
    """
    Load the folded 1D version of unquantized SpeechCom dataset for 35 classes.
    """
    return KWS_get_datasets(data, load_train, load_test, dataset_name='KWS_35',
                            num_classes=35, quantized=False)


def KWS_20_msnoise_mixed_get_datasets(data, load_train=True, load_test=True,
                                      apply_prob=0.8, snr_range=(-5, 10),
                                      desired_probs=None):
    """
    Returns the KWS dataset mixed with MSnoise dataset. Only training set will be mixed
    with MSnoise. Selected noise types will be applied to the training dataset using a randomly
    generated SNR value between the previously selected snr range. Noise will be applied to the
    training data using the application probability accordingly.

    Default parameter values are selected as:
    apply_prob --> 0.8 probability for application of additional noise on training data.
    snr_range  --> [-5, 10] dB SNR range to be used during the SNR selection for additional noise.
    noise_type --> All noise types in the noise dataset.
    """

    noise_type = MSnoise.class_dict

    if len(snr_range) > 1:
        snr_range = range(snr_range[0], snr_range[1])
    else:
        snr_range = list(snr_range)

    (data_dir, _) = data

    kws_train_dataset, kws_test_dataset = KWS_20_get_datasets(
        data, load_train, load_test)

    if load_train:
        noise_dataset_train = MSnoise(root=data_dir, classes=noise_type,
                                      d_type='train', dataset_len=len(kws_train_dataset),
                                      desired_probs=desired_probs,
                                      transform=None, quantize=False, download=True)

        train_dataset = SignalMixer(signal_dataset=kws_train_dataset,
                                    snr_range=snr_range,
                                    noise_type=noise_type, apply_prob=apply_prob,
                                    noise_dataset=noise_dataset_train)
    else:
        train_dataset = None

    if load_test:
        test_dataset = kws_test_dataset
    else:
        test_dataset = None

    return train_dataset, test_dataset


def KWS_12_benchmark_get_datasets(data, load_train=True, load_test=True):
    """
    Returns the KWS dataset benchmark for 12 classes. 10 keywords and
    _silence_ + _unknown_.
    """
    return KWS_get_datasets(data, load_train, load_test, dataset_name='KWS_12_benchmark',
                            num_classes=11, filter_libri=True, benchmark=True)


def MixedKWS_20_get_datasets_10dB(data, load_train=True, load_test=True,
                                  apply_prob=1, snr_range=tuple([10]),
                                  desired_probs=None):
    """
    Returns the mixed KWS dataset with MSnoise dataset under 10 dB SNR using signalmixer
    data loader. All of the training and test data will be augmented with
    additional noise.
    """

    noise_type = MSnoise.class_dict

    if len(snr_range) > 1:
        snr_range = range(snr_range[0], snr_range[1])
    else:
        snr_range = list(snr_range)

    (data_dir, _) = data

    kws_train_dataset, kws_test_dataset = KWS_20_get_datasets(
        data, load_train, load_test)

    if load_train:
        noise_dataset_train = MSnoise(root=data_dir, classes=noise_type,
                                      d_type='train', dataset_len=len(kws_train_dataset),
                                      desired_probs=desired_probs,
                                      transform=None, quantize=False, download=False)

        train_dataset = SignalMixer(signal_dataset=kws_train_dataset,
                                    snr_range=snr_range,
                                    noise_type=noise_type, apply_prob=apply_prob,
                                    noise_dataset=noise_dataset_train)
    else:
        train_dataset = None

    if load_test:
        noise_dataset_test = MSnoise(root=data_dir, classes=noise_type,
                                     d_type='test', dataset_len=len(kws_test_dataset),
                                     desired_probs=desired_probs,
                                     transform=None, quantize=False, download=False)

        test_dataset = SignalMixer(signal_dataset=kws_test_dataset,
                                   snr_range=snr_range,
                                   noise_type=noise_type, apply_prob=apply_prob,
                                   noise_dataset=noise_dataset_test)
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'KWS',  # 6 keywords + _unknown_
        'input': (512, 64),
        'output': KWS.dataset_dict['KWS'],
        'weight': (1, 1, 1, 1, 1, 1, 0.06),
        'loader': KWS_get_datasets,
    },
    {
        'name': 'KWS_20',  # 20 keywords + _unknown_
        'input': (128, 128),
        'output': KWS.dataset_dict['KWS_20'],
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.07),
        'loader': KWS_20_get_datasets,
    },
    {
        'name': 'KWS_35',  # 35 keywords + _unknown_
        'input': (128, 128),
        'output': KWS.dataset_dict['KWS_35'],
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.07),
        'loader': KWS_35_get_datasets,
    },
    {
        'name': 'KWS_20_msnoise_mixed',  # 20 keywords + _unknown_
        'input': (128, 128),
        'output': KWS.dataset_dict['KWS_20'],
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.07),
        'loader': KWS_20_msnoise_mixed_get_datasets,
    },
    {
        'name': 'KWS_12_benchmark',  # 10 keywords + _silence_ + _unknown_
        'input': (128, 128),
        'output': KWS.dataset_dict['KWS_12_benchmark'],
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.056),
        'loader': KWS_12_benchmark_get_datasets,
    },
    {
        'name': 'MixedKWS20_10dB',  # 20 keywords + _unknown_
        'input': (128, 128),
        'output': KWS.dataset_dict['KWS_20'],
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.07),
        'loader': MixedKWS_20_get_datasets_10dB,
    },
    {
        'name': 'KWS_35_unquantized',  # 35 keywords + _unknown_
        'input': (128, 128),
        'output': KWS.dataset_dict['KWS_35'],
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.07),
        'loader': KWS_35_get_unquantized_datasets,
    }
]
