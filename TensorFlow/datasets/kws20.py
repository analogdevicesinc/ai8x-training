###################################################################################################
#
# Copyright (C) 2018-2020 Maxim Integrated Products, Inc. All Rights Reserved.
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
Classes and functions used to create keyword spotting dataset.
"""
import errno
import hashlib
import os
import sys
import tarfile
import time
import warnings

import numpy as np

import librosa
from six.moves import urllib
from tqdm import tqdm


class KWS:
    """
    `SpeechCom v0.02 <http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz>`
    Dataset, 1D folded.

    Args:
    root (string): Root directory of dataset where ``KWS/processed/dataset.pt``
        exist.
    classes(array): List of keywords to be used.
    d_type(string): Option for the created dataset. ``train``, ``val``, ``test``.
    n_augment(int, optional): Number of augmented samples added to the dataset from
        each sample by random modifications, i.e. stretching, shifting and random noise.
    transform (callable, optional): A function/transform that takes in an PIL image
        and returns a transformed version.
    download (bool, optional): If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.

    """

    url = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
    fs = 16000

    class_dict = {
        'backward': 0,
        'bed': 1,
        'bird': 2,
        'cat': 3,
        'dog': 4,
        'down': 5,
        'eight': 6,
        'five': 7,
        'follow': 8,
        'forward': 9,
        'four': 10,
        'go': 11,
        'happy': 12,
        'house': 13,
        'learn': 14,
        'left': 15,
        'marvin': 16,
        'nine': 17,
        'no': 18,
        'off': 19,
        'on': 20,
        'one': 21,
        'right': 22,
        'seven': 23,
        'sheila': 24,
        'six': 25,
        'stop': 26,
        'three': 27,
        'tree': 28,
        'two': 29,
        'up': 30,
        'visual': 31,
        'wow': 32,
        'yes': 33,
        'zero': 34
    }

    desired_class_dict = {
        'up': 0,
        'down': 1,
        'left': 2,
        'right': 3,
        'stop': 4,
        'go': 5,
        'yes': 6,
        'no': 7,
        'on': 8,
        'off': 9,
        'one': 10,
        'two': 11,
        'three': 12,
        'four': 13,
        'five': 14,
        'six': 15,
        'seven': 16,
        'eight': 17,
        'nine': 18,
        'zero': 19
    }

    def __init__(self, root, transform=None, download=False):

        self.root = root
        self.transform = transform

        self.data_file = 'kws20_dataset.npz'

        print("Download:", download)
        if download:
            self.__download()

    @property
    def raw_folder(self):
        """Folder for the raw data.
        """
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        """Folder for the processed data.
        """
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def __download(self):

        if self.__check_exists():
            return

        self.__makedir_exist_ok(self.raw_folder)
        self.__makedir_exist_ok(self.processed_folder)

        filename = self.url.rpartition('/')[2]
        print("location: ", filename)
        self.__download_and_extract_archive(self.url, download_root=self.raw_folder,
                                            filename=filename)

        self.__gen_datasets()

    def __check_exists(self):
        return os.path.exists(
            os.path.join(self.processed_folder, self.data_file))

    def __makedir_exist_ok(self, dirpath):  # pylint: disable=no-self-use
        try:
            os.makedirs(dirpath)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

    def __gen_bar_updater(self):  # pylint: disable=no-self-use
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
                urllib.request.urlretrieve(
                    url, fpath, reporthook=self.__gen_bar_updater())
            except (urllib.error.URLError, IOError) as e:
                if url[:5] == 'https':
                    url = url.replace('https:', 'http:')
                    print('Failed download. Trying https -> http instead.'
                          ' Downloading ' + url + ' to ' + fpath)
                    urllib.request.urlretrieve(
                        url, fpath, reporthook=self.__gen_bar_updater())
                else:
                    raise e

    def __calculate_md5(self, fpath, chunk_size=1024 * 1024):  # pylint: disable=no-self-use
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

    def __extract_archive(  # pylint: disable=no-self-use
            self,
            from_path,
            to_path=None,
            remove_finished=False):
        if to_path is None:
            to_path = os.path.dirname(from_path)

        if from_path.endswith('.tar.gz'):
            with tarfile.open(from_path, 'r:gz') as tar:
                tar.extractall(path=to_path)
        else:
            raise ValueError(
                "Extraction of {} not supported".format(from_path))

        if remove_finished:
            os.remove(from_path)

    def __download_and_extract_archive(self,
                                       url,
                                       download_root,
                                       extract_root=None,
                                       filename=None,
                                       md5=None,
                                       remove_finished=False):
        download_root = os.path.expanduser(download_root)
        if extract_root is None:
            extract_root = download_root
        if not filename:
            filename = os.path.basename(url)

        self.__download_url(url, download_root, filename, md5)

        archive = os.path.join(download_root, filename)
        print("Extracting {} to {}".format(archive, extract_root))
        self.__extract_archive(archive, extract_root, remove_finished)

    def getfile(self):
        """
        gets the path to the processed dataset file
        """
        return os.path.join(self.processed_folder, self.data_file)

    @staticmethod
    def add_white_noise(audio, noise_var_coeff):
        """Adds zero mean Gaussian noise to image with specified variance.
        """
        coeff = noise_var_coeff * np.mean(np.abs(audio))
        noisy_audio = audio + coeff * np.random.randn(len(audio))
        return noisy_audio

    @staticmethod
    def shift(audio, shift_sec, fs):
        """Shifts audio.
        """
        shift_count = int(shift_sec * fs)
        return np.roll(audio, shift_count)

    @staticmethod
    def stretch(audio, rate=1):
        """Stretches audio with specified ratio.
        """
        input_length = 16000
        audio2 = librosa.effects.time_stretch(audio, rate)
        if len(audio2) > input_length:
            audio2 = audio2[:input_length]
        else:
            audio2 = np.pad(audio2, (0, max(0, input_length - len(audio2))),
                            "constant")

        return audio2

    def augment(self, audio, fs, verbose=False):
        """Augments audio by adding random noise, shift and stretch ratio.
        """
        random_noise_var_coeff = np.random.uniform(0, 1)
        random_shift_time = np.random.uniform(-0.1, 0.1)
        random_strech_coeff = np.random.uniform(0.8, 1.3)

        aug_audio = self.stretch(audio, random_strech_coeff)
        aug_audio = self.shift(aug_audio, random_shift_time, fs)
        aug_audio = self.add_white_noise(aug_audio, random_noise_var_coeff)
        if verbose:
            print(
                f'random_noise_var_coeff: {random_noise_var_coeff:.2f}\nrandom_shift_time: \
                    {random_shift_time:.2f}\nrandom_strech_coeff: {random_strech_coeff:.2f}'
            )
        return aug_audio

    def augment_multiple(self, audio, fs, n_augment, verbose=False):
        """Calls `augment` function for n_augment times for given audio data.
        Finally the original audio is added to have (n_augment+1) audio data.
        """
        aug_audio = [
            self.augment(audio, fs, verbose) for i in range(n_augment)
        ]
        aug_audio.insert(0, audio)
        return aug_audio

    @staticmethod
    def quantize_audio(data):
        """quantize audio to 8 bit
        """
        step_size = 2.0 / 256.0
        q_data = np.round((data - (-1.0)) / step_size)
        q_data = np.clip(q_data, 0, 255)
        return np.uint8(q_data)

    def __gen_datasets(self):
        print('Generating dataset from raw data samples for the first time. ')
        print('Warning: This process could take an hour!')
        with warnings.catch_warnings():
            warnings.simplefilter('error')

            test_data_path = self.raw_folder
            lst = os.listdir(test_data_path)
            lst = sorted(lst)
            labels = [
                d for d in lst
                if os.path.isdir(os.path.join(test_data_path, d))
                and d[0].isalpha()
            ]

            # PARAMETERS
            data_len = 128 * 128
            print('data_len: %s' % data_len)

            aug_num = 1  # No dataset augmentation

            if aug_num == 0:
                print('No augmentation added')
            else:
                print('Dataset is augmented by: ', aug_num)

            # show the size of dataset for each keyword
            print('------------- Label Size ---------------')
            for i, label in enumerate(labels):
                records = os.listdir(os.path.join(test_data_path, label))
                print('%8s:  \t%d' % (label, len(records)))
            print('------------------------------------------')

            # find the size of dataset
            size = 0
            for label in labels:
                size += (aug_num + 1) * len(
                    os.listdir(os.path.join(test_data_path, label)))

            print(f"total size of dataset:{size}")

            train_images = np.empty((size, 128, 128), dtype=np.uint8)
            valid_images = np.empty((size, 128, 128), dtype=np.uint8)
            test_images = np.empty((size, 128, 128), dtype=np.uint8)
            train_labels = np.empty((size), dtype=np.uint8)
            valid_labels = np.empty((size), dtype=np.uint8)
            test_labels = np.empty((size), dtype=np.uint8)

            tottraincount = 0
            totvalidcount = 0
            tottestcount = 0

            for i, label in enumerate(labels):

                print(
                    f'\nProcessing the label: {label}. {i + 1} of {len(labels)}'
                )

                # filtering based on desired labels
                if label in self.desired_class_dict:
                    class_id = self.desired_class_dict[label]
                    msg = 'in desired list, update label num:'
                else:
                    class_id = len(self.desired_class_dict)
                    msg = 'not in desired list, relabel as: unknown, label num:'
                print(
                    f'Current label: {label}, label num:{i} {msg} {class_id}')

                records = os.listdir(os.path.join(test_data_path, label))
                records = sorted(records)

                time1 = time.time()
                traincount = 0
                validatecount = 0
                testcount = 0

                for r, record in enumerate(records):

                    if r % 1000 == 0:
                        print('\t%d of %d' % (r + 1, len(records)))

                    if hash(record) % 10 < 8:
                        d_typ = np.uint8(0)  # train
                        traincount += 1
                    elif hash(record) % 10 < 9:
                        d_typ = np.uint8(1)  # val
                        validatecount += 1
                    else:
                        d_typ = np.uint8(2)  # test
                        testcount += 1

                    record_pth = os.path.join(test_data_path, label, record)
                    y, fs = librosa.load(record_pth, offset=0, sr=None)
                    audio_list = self.augment_multiple(
                        y, fs, aug_num, verbose=False)
                    for n_a, y in enumerate(audio_list):
                        # store set type: train, validate or test
                        assert n_a < 3

                        if y.size >= data_len:
                            y = y[:data_len]
                        else:
                            y = np.pad(y, [0, data_len - y.size], 'constant')

                        # Write audio 128x128=16384 samples without overlap
                        for j in range(128):
                            audio_seq1 = y[(j * 128):(j * 128 + 128)]

                            if d_typ == 0:  # train
                                train_images[
                                    tottraincount,
                                    j, :] = self.quantize_audio(  # No matrix transpose
                                        audio_seq1)
                                train_labels[tottraincount] = class_id

                            elif d_typ == 1:  # val
                                valid_images[
                                    totvalidcount,
                                    j, :] = self.quantize_audio(  # No matrix transpose
                                        audio_seq1)
                                valid_labels[totvalidcount] = class_id

                            elif d_typ == 2:  # test
                                test_images[
                                    tottestcount,
                                    j, :] = self.quantize_audio(  # No matrix transpose
                                        audio_seq1)
                                test_labels[tottestcount] = class_id
                            else:
                                print("Bad data type", d_typ)
                                sys.exit(0)

                        # increment the counter for selected type
                        if d_typ == 0:
                            tottraincount += 1
                        elif d_typ == 1:  # val
                            totvalidcount += 1
                        else:  # test
                            tottestcount += 1

                dur = time.time() - time1
                print('Done in %.3fsecs.' % dur)
                print('Training: %d,  Validation: %d, Test: %d' %
                      (traincount, validatecount, testcount))

                time1 = time.time()

            print('Final Training: %d,  Validation: %d, Test: %d' %
                  (tottraincount, totvalidcount, tottestcount))

            # adjust the size based on assigned data
            train_images = train_images[:tottraincount, :, :]
            train_labels = train_labels[:tottraincount].flatten()

            valid_images = valid_images[:totvalidcount, :, :]
            valid_labels = valid_labels[:totvalidcount].flatten()

            test_images = test_images[:tottestcount, :, :]
            test_labels = test_labels[:tottestcount].flatten()

            print('Training set shape:', train_images.shape)
            print('Training labels shape:', train_labels.shape)
            print('Validation set shape:', valid_images.shape)
            print('Validation labels shape:', valid_labels.shape)
            print('Test set shape:', test_images.shape)
            print('Test labels shape:', test_labels.shape)

            np.savez(
                os.path.join(self.processed_folder, self.data_file),
                train_images=train_images,
                train_labels=train_labels,
                valid_images=valid_images,
                valid_labels=valid_labels,
                test_images=test_images,
                test_labels=test_labels)

        print('Dataset created!')


class KWS_20(KWS):
    """
    `SpeechCom v0.02 <http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz>`
    Dataset, 1D folded.
    """

    def __str__(self):
        return self.__class__.__name__


def KWS_20_get_datasets(data_dir):
    """
    Load raw samples from "speech_commands_v0.02" dataset

    The dataset is loaded from the archive file, so the file is required for this version.

    The dataset originally includes 35 keywords. A dataset is formed with 21 classes which
    includes 20 of the original keywords and the rest of the
    dataset is used to form the last class, i.e class of the others.
    The dataset is split into training, validation and test sets. 80:10:10 training:validation:test
    split is used by default.

    Data is augmented to 3x duplicate data by randomly stretch, shift and randomly add noise where
    the stretching coefficient, shift amount and noise variance are randomly selected between
    0.8 and 1.3, -0.1 and 0.1, 0 and 1, respectively.
    """

    # download and store dataset in npz format
    dataset = KWS(root=data_dir, download=True)

    # get file location
    file = dataset.getfile()

    # load dataset file
    a = np.load(file)

    # read images, data range should be -128 to 127
    train_images = a['train_images'].astype(np.int32) - 128
    train_labels = a['train_labels']
    valid_images = a['valid_images'].astype(np.int32) - 128
    valid_labels = a['valid_labels']
    test_images = a['test_images'].astype(np.int32) - 128
    test_labels = a['test_labels']

    return (train_images, train_labels), (valid_images,
                                          valid_labels), (test_images,
                                                          test_labels)


def get_datasets(data_dir):
    """
    generic get the dataset in form of (train_images,train_labels), (valid_images, valid_labels),
    (test_images, test_labels)
    """
    return KWS_20_get_datasets(data_dir)


def get_classnames():
    """
    name of labels
    """
    class_names = [
        'up', 'down', 'left', 'right', 'stop', 'go', 'yes', 'no', 'on', 'off',
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
        'zero', 'unknown'
    ]
    return class_names
