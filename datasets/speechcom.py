###################################################################################################
#
# Copyright (C) 2019-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Classes and functions used to utilize Speech Commands dataset.
"""
import errno
import hashlib
import os
import tarfile
import urllib
import warnings

import numpy as np
import torch
from torch.utils.model_zoo import tqdm
from torchvision import transforms

import librosa
import librosa.display
from PIL import Image

import ai8x


class SpeechCom(torch.utils.data.Dataset):
    """`SpeechCom v0.02 <http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz>`
    Dataset.

    Args:
        root (string): Root directory of dataset where ``SpeechCom/processed/train.pt``
            ``SpeechCom/processed/val.pt`` and  ``SpeechCom/processed/test.pt`` exist.
        classes(array): List of keywords to be used.
        d_type(string): Option for the created dataset. ``train`` is to create dataset
            from ``training.pt``, ``val`` is to create dataset from ``val.pt``, ``test``
            is to create dataset from ``test.pt``.
        n_augment(int, optional): Number of samples added to the dataset from each sample
            by random modifications, i.e. stretching, shifting and random noise addition.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    url = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
    fs = 16000
    training_file = 'train.pt'
    test_file = 'test.pt'
    validation_file = 'val.pt'

    class_dict = {'backward': 0, 'bed': 1, 'bird': 2, 'cat': 3, 'dog': 4, 'down': 5,
                  'eight': 6, 'five': 7, 'follow': 8, 'forward': 9, 'four': 10, 'go': 11,
                  'happy': 12, 'house': 13, 'learn': 14, 'left': 15, 'marvin': 16, 'nine': 17,
                  'no': 18, 'off': 19, 'on': 20, 'one': 21, 'right': 22, 'seven': 23,
                  'sheila': 24, 'six': 25, 'stop': 26, 'three': 27, 'tree': 28, 'two': 29,
                  'up': 30, 'visual': 31, 'wow': 32, 'yes': 33, 'zero': 34}

    def __init__(self, root, classes, d_type, n_augment=0, transform=None, download=False):
        self.root = root
        self.classes = classes
        self.d_type = d_type
        self.transform = transform
        self.n_augment = n_augment

        if download:
            self.__download()

        if self.d_type == 'train':
            data_file = self.training_file
        elif self.d_type == 'test':
            data_file = self.test_file
        elif self.d_type == 'val':
            data_file = self.validation_file
        else:
            print(f'Unknown data type: {d_type}')
            return

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
        self.__filter_classes()

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

    def __gen_datasets(self):
        print('Generating dataset from raw data samples.')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            lst = os.listdir(self.raw_folder)
            labels = [d for d in lst if os.path.isdir(os.path.join(self.raw_folder, d)) and
                      d[0].isalpha()]
            train_images = []
            val_images = []
            test_images = []
            train_labels = []
            val_labels = []
            test_labels = []
            for i, label in enumerate(labels):
                print(f'\tProcessing the label: {label}. {i+1} of {len(labels)}')
                records = os.listdir(os.path.join(self.raw_folder, label))
                records = sorted(records)
                for record in records:
                    record_pth = os.path.join(self.raw_folder, label, record)
                    y, _ = librosa.load(record_pth, offset=0, sr=None)

                    audio_list = augment_multiple(audio=y, fs=self.fs, n_augment=self.n_augment)
                    for augmented_audio in audio_list:
                        S_8bit = audio2image(audio=augmented_audio, sr=self.fs, n_mels=64,
                                             f_max=8000, hop_length=256, n_fft=512)
                        if S_8bit is not None:
                            if hash(record) % 10 < 7:
                                train_images.append(S_8bit)
                                train_labels.append(label)
                            elif hash(record) % 10 < 9:
                                val_images.append(S_8bit)
                                val_labels.append(label)
                            else:
                                test_images.append(S_8bit)
                                test_labels.append(label)

            print(f'{silence_counter} of {total_counter} are rejected as no '
                  'keyword is detected in the record.')

            train_images = torch.from_numpy(np.array(train_images))
            val_images = torch.from_numpy(np.array(val_images))
            test_images = torch.from_numpy(np.array(test_images))

            label_dict = dict(zip(labels, range(35)))
            train_labels = torch.from_numpy(np.array([label_dict[ll] for ll in train_labels]))
            val_labels = torch.from_numpy(np.array([label_dict[ll] for ll in val_labels]))
            test_labels = torch.from_numpy(np.array([label_dict[ll] for ll in test_labels]))

            train_set = (train_images, train_labels)
            val_set = (val_images, val_labels)
            test_set = (test_images, test_labels)

            torch.save(train_set, os.path.join(self.processed_folder, self.training_file))
            torch.save(val_set, os.path.join(self.processed_folder, self.validation_file))
            torch.save(test_set, os.path.join(self.processed_folder, self.test_file))

        print('Dataset created!')

    def __download(self):
        if self.__check_exists():
            return

        self.__makedir_exist_ok(self.raw_folder)
        self.__makedir_exist_ok(self.processed_folder)

        filename = self.url.rpartition('/')[2]
        self.__download_and_extract_archive(self.url, download_root=self.raw_folder,
                                            filename=filename)
        self.__gen_datasets()

    def __check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) |
                os.path.exists(os.path.join(self.processed_folder, self.test_file)) |
                os.path.exists(os.path.join(self.processed_folder, self.validation_file)))

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
                tar.extractall(path=to_path)
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

    def __filter_classes(self):
        initial_new_class_label = len(self.class_dict)
        new_class_label = initial_new_class_label
        for c in self.classes:
            if c not in self.class_dict:
                print(f'Class is not in the data: {c}')
                return
            # else:
            print(f'Class {c}, {self.class_dict[c]}')
            num_elems = (self.targets == self.class_dict[c]).cpu().sum()
            print(f'Number of elements in class {c}: {num_elems}')
            self.targets[(self.targets == self.class_dict[c])] = new_class_label
            new_class_label += 1

        num_elems = (self.targets < initial_new_class_label).cpu().sum()
        print(f'Number of elements in class unknown: {num_elems}')
        self.targets[(self.targets < initial_new_class_label)] = new_class_label
        self.targets -= initial_new_class_label
        print(np.unique(self.targets.data.cpu()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class SpeechCom_20(SpeechCom):
    """
    `SpeechCom v0.02 <http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz>`
    Dataset, 1D folded.
    """


# functions to convert audio data to image by mel spectrogram technique and augment data.
silence_counter = 0
total_counter = 0


def audio2image(audio, sr, n_mels, f_max, hop_length, n_fft):
    """Converts audio to an image form by taking mel spectrogram.
    """
    global silence_counter  # pylint: disable=global-statement
    global total_counter  # pylint: disable=global-statement

    total_counter += 1
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmax=f_max,
                                       hop_length=hop_length, n_fft=n_fft)
    try:
        S = np.maximum(10*np.log10(10e-13 + S / np.max(S)), -64)
    except (ZeroDivisionError, ValueError):
        return None

    S_8bit = (4*(S + 64) - 10e-13) // 1
    S_8bit = np.maximum(S_8bit, 0)

    if np.std(np.mean(S, axis=0)) < 1.2:
        silence_counter += 1
        return None

    S_8bit = (np.hstack((S_8bit, np.zeros((64, 64-S_8bit.shape[1]))))).astype(np.uint8)
    return S_8bit


def load_audio_file(file_path):
    """Loads audio data from specified file location.
    """
    input_length = 16000
    audio = librosa.core.load(file_path)[0]  # sr=16000
    if len(audio) > input_length:
        audio = audio[:input_length]
    else:
        audio = np.pad(audio, (0, max(0, input_length - len(audio))), "constant")
    return audio


def add_white_noise(audio, noise_var_coeff):
    """Adds zero mean Gaussian noise to image with specified variance.
    """
    coeff = noise_var_coeff * np.mean(np.abs(audio))
    noisy_audio = audio + coeff * np.random.randn(len(audio))
    return noisy_audio


def shift(audio, shift_sec, fs):
    """Shifts audio.
    """
    shift_count = int(shift_sec * fs)
    return np.roll(audio, shift_count)


def stretch(audio, rate=1):
    """Stretchs audio with specified ratio.
    """
    input_length = 16000
    audio2 = librosa.effects.time_stretch(audio, rate)
    if len(audio2) > input_length:
        audio2 = audio2[:input_length]
    else:
        audio2 = np.pad(audio2, (0, max(0, input_length - len(audio2))), "constant")

    return audio2


def augment(audio, fs, verbose=False):
    """Augments audio by adding random noise, shift and stretch ratio.
    """
    random_noise_var_coeff = np.random.uniform(0, 1)
    random_shift_time = np.random.uniform(-0.1, 0.1)
    random_strech_coeff = np.random.uniform(0.8, 1.3)

    aug_audio = stretch(audio, random_strech_coeff)
    aug_audio = shift(aug_audio, random_shift_time, fs)
    aug_audio = add_white_noise(aug_audio, random_noise_var_coeff)
    if verbose:
        print(f'random_noise_var_coeff: {random_noise_var_coeff:.2f}\nrandom_shift_time: \
                {random_shift_time:.2f}\nrandom_strech_coeff: {random_strech_coeff:.2f}')
    return aug_audio


def augment_multiple(audio, fs, n_augment, verbose=False):
    """Calls `augment` function for n_augment times for given audio data.
    Finally the original audio is added to have (n_augment+1) audio data.
    """
    aug_audio = [augment(audio, fs, verbose) for i in range(n_augment)]
    aug_audio.insert(0, audio)
    return aug_audio


def speechcom_get_datasets(data, load_train=True, load_test=True, num_classes=6):
    """
    Load the SpeechCom v0.02 dataset
    (https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz).

    The dataset originally includes 30 keywords. A dataset is formed with 7 classes which includes
    6 of the original keywords ('up', 'down', 'left', 'right', 'stop', 'go') and the rest of the
    dataset is used to form the last class, i.e class of the others.
    The dataset is split into training, validation and test sets. 80:10:10 training:validation:test
    split is used by default.

    Data is augmented to 5x duplicate data by randomly stretch, shift and randomly add noise where
    the stretching coefficient, shift amount and noise variance are randomly selected between
    0.8 and 1.3, -0.1 and 0.1, 0 and 1, respectively.
    """
    (data_dir, args) = data

    if num_classes == 6:
        classes = ['up', 'down', 'left', 'right', 'stop', 'go']  # 6 keywords
    elif num_classes == 20:
        classes = ['up', 'down', 'left', 'right', 'stop', 'go', 'yes', 'no', 'on', 'off', 'one',
                   'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero']
    else:
        raise ValueError(f'Unsupported num_classes {num_classes}')

    transform = transforms.Compose([
        transforms.ToTensor(),
        ai8x.normalize(args=args)
    ])

    if load_train:
        train_dataset = SpeechCom(root=data_dir, classes=classes, d_type='train', n_augment=4,
                                  transform=transform, download=True)
    else:
        train_dataset = None

    if load_test:
        test_dataset = SpeechCom(root=data_dir, classes=classes, d_type='val', n_augment=4,
                                 transform=transform, download=True)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


def speechcom_20_get_datasets(data, load_train=True, load_test=True):
    """
    Load the SpeechCom v0.02 dataset

    The dataset is loaded from the archive file, so the file is required for this version.

    The dataset originally includes 30 keywords. A dataset is formed with 21 classes which includes
    20 of the original keywords and the rest of the dataset is used to form the last class, i.e.,
    class of the others.
    The dataset is split into training, validation and test sets. 80:10:10 training:validation:test
    split is used by default.

    Data is augmented to 3x duplicate data by randomly stretch, shift and randomly add noise where
    the stretching coefficient, shift amount and noise variance are randomly selected between
    0.8 and 1.3, -0.1 and 0.1, 0 and 1, respectively.
    """
    return speechcom_get_datasets(data, load_train, load_test, num_classes=20)


datasets = [
    {
        'name': 'SpeechCom',  # 6 keywords
        'input': (1, 64, 64),
        'output': (0, 1, 2, 3, 4, 5, 6),
        'weight': (1, 1, 1, 1, 1, 1, 0.06),
        'loader': speechcom_get_datasets,
    },
    {
        'name': 'SpeechCom_20',  # 20 keywords
        'input': (1, 64, 64),
        'output': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
        'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.14),
        'loader': speechcom_20_get_datasets,
    },
]
