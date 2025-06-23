###################################################################################################
#
# Copyright (C) 2025 Analog Devices, Inc. All Rights Reserved.
# This software is proprietary to Analog Devices, Inc. and its licensors.
#
###################################################################################################
"""
Classes and functions used to create spectrumsense dataset.
"""
import errno
import glob
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib
import urllib.error
import urllib.request

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

import ai8x


class SpectrumSense(Dataset):
    """
    Dataloader for the Matlab Spectrum Sensing 5G/LTE Data Set/

    The dataset is provided by Mathworks and accompanies their
    'Spectrum Sensing with Deep Learning to Identify 5G and LTE Signals'
    which can be found at the following URL.
    https://www.mathworks.com/help/comm/ug/spectrum-sensing-with-deep-learning-to-identify-5g-and-lte-signals.html

    Possible class selections:
    0: None
    1: LTE
    2: NR
    """

    class_dict = {'None': 0, 'LTE': 1, 'NR': 2}

    # Values in the label images mapped to names
    mask_map = {'None': 0, 'LTE': 255, 'NR': 127}

    # pylint: disable-next=line-too-long
    url_spectrumsense = 'https://www.mathworks.com/supportfiles/spc/SpectrumSensing/SpectrumSenseTrainingDataNetwork.tar.gz'  # noqa: E501

    def __init__(self, root_dir, d_type, transform=None,
                 im_size=(256, 256), download=True):
        self.transform = transform
        self.root = root_dir

        # Download the data set if necessary
        if download:
            self.__download()

        if d_type not in ('test', 'train'):
            raise ValueError('Data type name msut be test or train.')

        src_folder = os.path.join(self.root, d_type)
        lbl_folder = os.path.join(self.root, d_type + '_labels')
        self.src_list = []
        self.lbl_list = []

        self.__create_mask_dict(im_size)

        for src_file in sorted(glob.glob('*.png', root_dir=src_folder, recursive=False)):
            lbl_file = os.path.splitext(src_file)[0] + '_lbl.png'

            src_img = Image.open(os.path.join(src_folder, src_file))
            lbl_img = Image.open(os.path.join(lbl_folder, lbl_file))

            # Dataset images are 256x256. Pad out when the image size is 352
            if im_size == [352, 352]:
                (src_img, lbl_img) = self.pad_image_and_label(src_img, lbl_img, im_size)

            src_data = SpectrumSense.normalize(np.asarray(src_img).astype(np.float32))

            # Convert the grayscale label image to 3 channel (RGB)
            lbl_gray = np.asarray(lbl_img)
            lbl_rgb = np.empty((lbl_gray.shape[0], lbl_gray.shape[1], 3))
            for i in range(3):
                lbl_rgb[:, :, i] = lbl_gray
            lbl_data = np.zeros((lbl_rgb.shape[0], lbl_rgb.shape[1]), dtype=np.uint8)

            for label_idx, (_, mask) in enumerate(self.label_mask_dict.items()):
                res = lbl_rgb == mask
                res = (label_idx) * res.all(axis=2)
                lbl_data += res.astype(np.uint8)

            self.src_list.append(src_data)
            self.lbl_list.append(lbl_data)

    def __create_mask_dict(self, img_dims):
        self.label_mask_dict = {}
        for lbl in self.class_dict:
            val = self.mask_map[lbl]
            label_mask = np.zeros((img_dims[0], img_dims[1], 3), dtype=np.uint8)
            label_mask[:, :, 0] = np.uint8(val)
            label_mask[:, :, 1] = np.uint8(val)
            label_mask[:, :, 2] = np.uint8(val)
            self.label_mask_dict[lbl] = label_mask

    @classmethod
    def pad_image_and_label(cls, img, lbl, im_size):
        """
        Pads the original images to the specified size and
        pastes the original content at the origin
        """

        img_pad = Image.new(img.mode, im_size)
        img_pad.paste(img, (0, 0))

        lbl_pad = Image.new(lbl.mode, im_size)
        lbl_pad.paste(lbl, (0, 0))
        return (img_pad, lbl_pad)

    @staticmethod
    def normalize(data):
        """Normalizes data to the range [0, 1)"""
        return data / 255.

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, idx):
        if self.transform is not None:
            img = self.transform(self.src_list[idx])
        return img, self.lbl_list[idx].astype(np.int64)

    def __download(self):
        train_dir = os.path.join(self.root, 'train')
        test_dir = os.path.join(self.root, 'test')
        train_lbl_dir = os.path.join(self.root, 'train_labels')
        test_lbl_dir = os.path.join(self.root, 'test_labels')

        self.__makedir_exist_ok(train_dir)
        self.__makedir_exist_ok(test_dir)
        self.__makedir_exist_ok(train_lbl_dir)
        self.__makedir_exist_ok(test_lbl_dir)

        if self.__check_data_exists():
            return  # skip do

        # Need to make sure the user has the hdftor8 utility to convert Matlab
        # HDF4 format to a PNG.  The pyHDF library didn't seem to expose the
        # DFR8 API
        if shutil.which('hdftor8') is None:
            print('This data set requires the HDF4 Tools to be installed\n'
                  'Please use apt-get install hdf4-tools, or the respective '
                  'command for your platform')
            raise RuntimeError('HDF4 Tools Not installed.')

        # Do everything in temporary directory that can be cleaned up later
        tmpdir = tempfile.mkdtemp()
        extract_dir = os.path.join(tmpdir, 'rawdata')
        filename = self.url_spectrumsense.rpartition('/')[2]
        self.__download_url(self.url_spectrumsense, tmpdir, filename)
        self.__extract_archive(os.path.join(tmpdir, filename), extract_dir)

        # Copy the Mathworks licenses over
        print('This dataset is subject the license terms as described by Mathworks.\n'
              'Please review the license files extracted from the archive.')
        lic_files = list(glob.glob('*license*', root_dir=extract_dir, recursive=False))
        for lic in lic_files:
            shutil.copy(os.path.join(extract_dir, lic), self.root)

        # Process the data files. TrainingData is used for training
        processed_files = self.__process_data(
            os.path.join(extract_dir, 'TrainingData'), train_dir, train_lbl_dir)
        # The LTE_NR sub-folder is used for test
        processed_files += self.__process_data(
            os.path.join(extract_dir, 'TrainingData', 'LTE_NR'), test_dir, test_lbl_dir)

        # Write all the files processed to manifest.txt. This doesn't really add
        # any value, but we'll use it to confirm this entire process was
        # successful next time training is requested.
        with open(os.path.join(self.root, 'manifest.txt'), 'w', encoding='utf-8') as mf:
            mf.write('\n'.join(sorted(processed_files)))

        # Clean up downloaded archive and files
        shutil.rmtree(tmpdir)

    def __process_data(self, from_path, to_path, to_lbl_path):
        """
        Processes all the PNG files from the from_path and converts their
        respective HDF files to grayscale PNGs
        """
        # Find all the PNGs. Don't recurse as a sub folder may be used for
        # different train/test data sets
        img_files = list(glob.glob('*.png', root_dir=from_path, recursive=False))
        proc_files = []

        for img in img_files:
            # Grab the original source image
            shutil.copy(os.path.join(from_path, img), to_path)

            # Create a temp file. Convert the HDF to that temp file. This will
            # just be a byte stream of data 65536 (256x256)
            rast_file = tempfile.mktemp()
            hdf_file = os.path.join(from_path, os.path.splitext(img)[0] + '.hdf')
            subprocess.run('hdftor8 ' + hdf_file + ' -r ' + rast_file, shell=True, check=False)

            # Now open the file as binary, and tell PIL to treat it like a
            # 256x256 8-bit single channel ('L')
            result_file = os.path.splitext(img)[0] + '_lbl.png'
            with open(rast_file, 'rb') as f:
                im = Image.frombytes('L', (256, 256), f.read())
                im.save(os.path.join(to_lbl_path, result_file))
            os.remove(rast_file)  # Clean up

            proc_files.append(img)
            proc_files.append(result_file)
        return proc_files

    def __extract_archive(self, from_path, to_path=None, remove_finished=False):
        if to_path is None:
            to_path = os.path.dirname(from_path)

        if from_path.endswith('.tar.gz'):
            with tarfile.open(from_path, 'r:gz') as tar:
                tar.extractall(path=to_path)
        else:
            raise ValueError(f"Extraction of {from_path} not supported")

        if remove_finished:
            os.remove(from_path)

    def __download_url(self, url, dl_dir, filename=None):
        if not filename:
            filename = os.path.basename(url)
        fpath = os.path.join(dl_dir, filename)
        self.__makedir_exist_ok(dl_dir)

        # downloads file
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)
            else:
                raise e

    def __check_data_exists(self):
        # Check for manifest.txt to determine if was already done
        return os.path.exists(os.path.join(self.root, 'manifest.txt'))

    def __makedir_exist_ok(self, dirpath):
        try:
            os.makedirs(dirpath)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise


def spectrumsense_get_datasets_s256(data, load_train=True, load_test=True):
    """
    Load the spectrumsense dataset in 48x88x88 format which are composed of 3x352x352 images folded
    with a fold_ratio of 4.
    """
    (data_dir, args) = data

    transform = transforms.Compose([transforms.ToTensor(),
                                    ai8x.normalize(args=args),
                                    ai8x.fold(fold_ratio=4)])

    if load_train:
        train_dataset = SpectrumSense(root_dir=os.path.join(data_dir, 'SpectrumSense'),
                                      d_type='train',
                                      im_size=[256, 256],
                                      transform=transform)
    else:
        train_dataset = None

    if load_test:
        test_dataset = SpectrumSense(root_dir=os.path.join(data_dir, 'SpectrumSense'),
                                     d_type='test',
                                     im_size=[256, 256],
                                     transform=transform)
        if args.truncate_testset:
            test_dataset.src_list = test_dataset.src_list[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


def spectrumsense_get_datasets_s352(data, load_train=True, load_test=True):
    """
    Load the spectrumsense dataset in 48x88x88 format which are composed of 3x352x352 images folded
    with a fold_ratio of 4.
    """
    (data_dir, args) = data

    transform = transforms.Compose([transforms.ToTensor(),
                                    ai8x.normalize(args=args),
                                    ai8x.fold(fold_ratio=4)])

    if load_train:
        train_dataset = SpectrumSense(root_dir=os.path.join(data_dir, 'SpectrumSense'),
                                      d_type='train',
                                      im_size=[352, 352],
                                      transform=transform)
    else:
        train_dataset = None

    if load_test:
        test_dataset = SpectrumSense(root_dir=os.path.join(data_dir, 'SpectrumSense'),
                                     d_type='test',
                                     im_size=[352, 352],
                                     transform=transform)
        if args.truncate_testset:
            test_dataset.src_list = test_dataset.src_list[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


def spectrumsense_get_datasets_s256_c3(data, load_train=True, load_test=True):
    """
    Load the spectrumsense dataset for 3 classes in 48x64x64 images.
    """
    return spectrumsense_get_datasets_s256(data, load_train, load_test)


def spectrumsense_get_datasets_s352_c3(data, load_train=True, load_test=True):
    """
    Load the spectrumsense dataset for 3 classes in 48x88x88 images.
    """
    return spectrumsense_get_datasets_s352(data, load_train, load_test)


datasets = [
    {
        'name': 'SpectrumSense_s256_c2',  # 2 classes
        'input': (48, 64, 64),
        'output': (0, 1, 2),
        'weight': (1, 1, 1),
        'loader': spectrumsense_get_datasets_s256_c3,
    },
    {
        'name': 'SpectrumSense_s352_c2',  # 2 classes
        'input': (48, 88, 88),
        'output': (0, 1, 2),
        'weight': (1, 1, 1),
        'loader': spectrumsense_get_datasets_s352_c3,
        'fold_ratio': 4,
    }
]
