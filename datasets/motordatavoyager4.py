###################################################################################################
#
# Copyright (C) 2024 Analog Devices, Inc. All Rights Reserved.
# This software is proprietary to Analog Devices, Inc. and its licensors.
#
###################################################################################################
"""
Classes and functions for the Sample Voyager4 Dataset
"""
import os
import pickle

import numpy as np
import torch
from numpy.fft import fft
from torch.utils.data import Dataset
from torchvision import transforms

import pandas as pd

import ai8x
from datasets import cbm_dataframe_parser
from utils.dataloader_utils import makedir_exist_ok


class MotorDataVoyager4(Dataset):  # pylint: disable=too-many-instance-attributes
    """
    Sample data loader file for AC Fan Test Data Files
    """

    # ADXL382 has a sampling frequency of 16kHz
    sampling_rate_in_Hz_ADXL382 = 16000

    faulty_files_folder = 'FaultyData'
    normal_files_folder = 'NormalData'

    # 80% of normal ata will be split into training
    # 20% of normal data will be validation set in training mode
    # 20% normal data + anomaly data will be testset in evaluation mode
    train_ratio = 0.8

    # Data will have dimension: (cnn_1dinput_len, 3)
    cnn_1dinput_len = 256

    # Raw data window length calculation:
    downsampling_ratio = 4
    raw_data_window_len = cnn_1dinput_len * downsampling_ratio * 2  # 2048
    raw_data_window_duration_in_sec = \
        raw_data_window_len / sampling_rate_in_Hz_ADXL382  # 0.128 sec

    overlap_ratio = 0.75

    def __init__(self, root, d_type, transform, eval_mode, label_as_signal,
                 speed_and_load_available):

        if d_type not in ('test', 'train'):
            raise ValueError("d_type can only be set to 'test' or 'train'")

        self.root = root
        self.d_type = d_type
        self.transform = transform
        self.speed_and_load_available = speed_and_load_available

        self.eval_mode = eval_mode
        self.label_as_signal = label_as_signal

        self.processed_folder = \
            os.path.join(root, self.__class__.__name__, 'processed')

        makedir_exist_ok(self.processed_folder)

        self.specs_identifier = f'eval_mode_{self.eval_mode}_' + \
                                f'label_as_signal_{self.label_as_signal}'

        train_dataset_pkl_file_path = \
            os.path.join(self.processed_folder, f'train_{self.specs_identifier}.pkl')

        test_dataset_pkl_file_path =  \
            os.path.join(self.processed_folder, f'test_{self.specs_identifier}.pkl')

        if self.d_type == 'train':
            self.dataset_pkl_file_path = train_dataset_pkl_file_path

        elif self.d_type == 'test':
            self.dataset_pkl_file_path = test_dataset_pkl_file_path

        self.signal_list = []
        self.lbl_list = []
        self.speed_list = []
        self.load_list = []

        self.__create_pkl_files()
        self.is_truncated = False

    @staticmethod
    def normalize_vib_features(features):
        """
        Normalize signal with Local Min Max Normalization
        """
        # Normalize data:
        for instance in range(features.shape[0]):
            instance_max = np.max(features[instance, :, :], axis=1)
            instance_min = np.min(features[instance, :, :], axis=1)

            for feature in range(features.shape[1]):
                for signal in range(features.shape[2]):
                    features[instance, feature, signal] = (
                        (features[instance, feature, signal] - instance_min[feature]) /
                        (instance_max[feature] - instance_min[feature])
                    )

        return features

    @staticmethod
    def process_raw_data_windows(file_raw_data_windows):
        """
        Processes each window of raw data
        """
        # First dimension: 3
        # Second dimension: number of windows
        # Third dimension: Window for self.duration_in_sec.
        num_features = file_raw_data_windows.shape[0]
        num_windows = file_raw_data_windows.shape[1]

        fft_output_window_size = int(MotorDataVoyager4.raw_data_window_len / 2)

        assert MotorDataVoyager4.raw_data_window_len == (MotorDataVoyager4.cnn_1dinput_len *
                                                         MotorDataVoyager4.downsampling_ratio * 2)

        file_cnn_signals = np.zeros((num_features, num_windows, MotorDataVoyager4.cnn_1dinput_len))

        # Perform FFT on each window () for each feature
        for window in range(num_windows):
            for feature in range(num_features):

                signal_for_fft = file_raw_data_windows[feature, window, :]

                fft_out = abs(fft(signal_for_fft))
                fft_out = fft_out[:fft_output_window_size]

                # downsample by taking average of
                # MotorDataVoyager4.downsampling_ratio (4) samples
                fft_out = np.average(fft_out.reshape(-1, MotorDataVoyager4.downsampling_ratio),
                                     axis=1)

                assert fft_out.shape[0] == MotorDataVoyager4.cnn_1dinput_len

                file_cnn_signals[feature, window, :] = fft_out

        # Reshape from (num_features, num_windows, window_size) into:
        # (num_windows, num_features, window_size)
        file_cnn_signals = file_cnn_signals.transpose([1, 0, 2])

        return file_cnn_signals

    @staticmethod
    def parse_ADXL382_and_return_raw_data(file_full_path):
        """
        Dataframe parser for Sample Voyager4 Motor Data csv file.
        Reads csv file and returns raw data
        The raw data size must be consecutive and bigger than window size.
        """
        df_raw = pd.read_csv(file_full_path, sep=',', usecols=['x', 'y', 'z'])

        raw_data = df_raw.to_numpy()

        window_size_assert_message = "CNN input length is incorrect."
        assert MotorDataVoyager4.raw_data_window_len <= \
            raw_data.shape[0], window_size_assert_message

        return raw_data

    def __create_pkl_files(self):
        if os.path.exists(self.dataset_pkl_file_path):

            print('\nPickle files are already generated ...\n')

            (self.signal_list, self.lbl_list, self.speed_list, self.load_list) = \
                pickle.load(open(self.dataset_pkl_file_path, 'rb'))
            return

        self.__gen_datasets()  # type: ignore

    def __gen_datasets(self):

        print('\nGenerating data frame pickle files from the raw data \n')

        healthy_data_dir = os.path.join(self.root,
                                        self.__class__.__name__,
                                        MotorDataVoyager4.normal_files_folder)
        faulty_data_dir = os.path.join(self.root,
                                       self.__class__.__name__,
                                       MotorDataVoyager4.faulty_files_folder)
        if not os.path.isdir(healthy_data_dir):
            raise FileNotFoundError(
                f'\nHealthy dataset directory {healthy_data_dir} does not exist.\n')

        if not os.path.isdir(faulty_data_dir):
            raise FileNotFoundError(
                f'\nFaulty dataset directory {faulty_data_dir} does not exist.\n')

        with os.scandir(healthy_data_dir) as it:
            if not any(it):
                raise FileNotFoundError(
                    f'\nHealthy dataset directory {healthy_data_dir} is empty.\n')

        with os.scandir(faulty_data_dir) as it:
            if not any(it):
                raise FileNotFoundError(
                    f'\nFaulty dataset directory {faulty_data_dir} is empty.\n')

        # Generate Normal Features:
        train_features = []
        train_speeds = []
        train_loads = []

        test_normal_features = []
        test_normal_speeds = []
        test_normal_loads = []

        for file in sorted(os.listdir(healthy_data_dir)):
            full_path = os.path.join(healthy_data_dir, file)

            # Users can parse raw data information and keep vibration speed and load properties
            file_speed = 0
            file_load = 0

            healthy_raw_data = \
                MotorDataVoyager4.parse_ADXL382_and_return_raw_data(full_path)

            healthy_raw_data_windows = \
                cbm_dataframe_parser.CbM_DataFrame_Parser.split_file_raw_data(
                    healthy_raw_data,
                    MotorDataVoyager4.sampling_rate_in_Hz_ADXL382,
                    MotorDataVoyager4.raw_data_window_duration_in_sec,
                    MotorDataVoyager4.overlap_ratio)

            file_cnn_signals = \
                MotorDataVoyager4.process_raw_data_windows(healthy_raw_data_windows)

            num_training = int(self.train_ratio * file_cnn_signals.shape[0])

            for i in range(file_cnn_signals.shape[0]):
                if i < num_training:
                    train_features.append(file_cnn_signals[i])
                    train_speeds.append(file_speed)
                    train_loads.append(file_load)
                else:
                    test_normal_features.append(file_cnn_signals[i])
                    test_normal_speeds.append(file_speed)
                    test_normal_loads.append(file_load)

        train_features = np.asarray(train_features)
        test_normal_features = np.asarray(test_normal_features)

        # Generate Anomaly Features:
        anomaly_features = []
        test_anormal_speeds = []
        test_anormal_loads = []

        for file in sorted(os.listdir(faulty_data_dir)):
            full_path = os.path.join(faulty_data_dir, file)

            # Users can parse raw data information and keep vibration speed and load properties
            file_speed = 0
            file_load = 0

            faulty_raw_data = \
                MotorDataVoyager4.parse_ADXL382_and_return_raw_data(full_path)

            faulty_raw_data_windows = \
                cbm_dataframe_parser.CbM_DataFrame_Parser.split_file_raw_data(
                    faulty_raw_data,
                    MotorDataVoyager4.sampling_rate_in_Hz_ADXL382,
                    MotorDataVoyager4.raw_data_window_duration_in_sec,
                    MotorDataVoyager4.overlap_ratio)

            file_cnn_signals = \
                MotorDataVoyager4.process_raw_data_windows(faulty_raw_data_windows)

            for i in range(file_cnn_signals.shape[0]):
                anomaly_features.append(file_cnn_signals[i])
                test_anormal_speeds.append(file_speed)
                test_anormal_loads.append(file_load)

        anomaly_features = np.asarray(anomaly_features)

        # Normalization
        train_features = MotorDataVoyager4.normalize_vib_features(train_features)
        test_normal_features = MotorDataVoyager4.normalize_vib_features(test_normal_features)
        anomaly_features = MotorDataVoyager4.normalize_vib_features(anomaly_features)

        # ARRANGE TEST-TRAIN SPLIT AND LABELS
        if self.d_type == 'train':
            self.lbl_list = [train_features[i, :, :] for i in range(train_features.shape[0])]
            self.signal_list = [torch.Tensor(label) for label in self.lbl_list]
            self.lbl_list = list(self.signal_list)
            self.speed_list = np.array(train_speeds)
            self.load_list = np.array(train_loads)

            if not self.label_as_signal:
                self.lbl_list = np.zeros([len(self.signal_list), 1])

        elif self.d_type == 'test':

            # Testing in training phase includes only normal test samples
            if not self.eval_mode:
                test_data = test_normal_features
            else:
                test_data = np.concatenate((test_normal_features, anomaly_features), axis=0)

            self.lbl_list = [test_data[i, :, :] for i in range(test_data.shape[0])]
            self.signal_list = [torch.Tensor(label) for label in self.lbl_list]
            self.lbl_list = list(self.signal_list)
            self.speed_list = np.concatenate((np.array(test_normal_speeds),
                                              np.array(test_anormal_speeds)))
            self.load_list = np.concatenate((np.array(test_normal_loads),
                                             np.array(test_anormal_loads)))

            if not self.label_as_signal:
                self.lbl_list = np.concatenate(
                                    (np.zeros([len(test_normal_features), 1]),
                                     np.ones([len(anomaly_features), 1])), axis=0)

        # Save pickle file
        pickle.dump((self.signal_list, self.lbl_list, self.speed_list, self.load_list),
                    open(self.dataset_pkl_file_path, 'wb'))

    def __len__(self):
        if self.is_truncated:
            return 1
        return len(self.signal_list)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError

        if self.is_truncated:
            index = 0

        signal = self.signal_list[index]

        # Model takes 256 channels! From 3, 256 to 256, 3
        signal = np.transpose(signal, (1, 0))

        lbl = self.lbl_list[index]

        if self.label_as_signal:
            lbl = np.transpose(lbl, (1, 0))

        if self.transform is not None:
            signal = self.transform(signal)

            if self.label_as_signal:
                lbl = self.transform(lbl)

        if not self.label_as_signal:
            lbl = lbl.astype(np.int64)  # type: ignore
        else:
            lbl = lbl.numpy().astype(np.float32)  # type: ignore

        if self.speed_and_load_available:
            speed = self.speed_list[index]
            load = self.load_list[index]

            return signal, lbl, speed, load

        return signal, lbl


def motordatavoyager4_get_datasets(data, load_train=True, load_test=True,
                                   eval_mode=False,
                                   label_as_signal=True,
                                   speed_and_load_available=False):
    """
    Returns Sample Motor Data acquired by Voyager4
    """
    (data_dir, args) = data

    if load_train:
        train_transform = transforms.Compose([
            ai8x.normalize(args=args)
        ])

        train_dataset = MotorDataVoyager4(root=data_dir, d_type='train',
                                          transform=train_transform,
                                          eval_mode=eval_mode,
                                          label_as_signal=label_as_signal,
                                          speed_and_load_available=speed_and_load_available)

        print(f'Train dataset length: {len(train_dataset)}\n')
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            ai8x.normalize(args=args)
        ])

        test_dataset = MotorDataVoyager4(root=data_dir, d_type='test',
                                         transform=test_transform,
                                         eval_mode=eval_mode,
                                         label_as_signal=label_as_signal,
                                         speed_and_load_available=speed_and_load_available)

        print(f'Test dataset length: {len(test_dataset)}\n')
    else:
        test_dataset = None

    return train_dataset, test_dataset


def motordatavoyager4_get_datasets_for_train(data,
                                             load_train=True,
                                             load_test=True):
    """"
    Returns Sample Voyager4 Dataset For Training Mode:

    Trainset samples include 80% normal samples and
    validation set includes only normals (remaining 20%)
    """

    eval_mode = False
    label_as_signal = True

    speed_and_load_available = False

    return motordatavoyager4_get_datasets(data, load_train, load_test,
                                          eval_mode=eval_mode,
                                          label_as_signal=label_as_signal,
                                          speed_and_load_available=speed_and_load_available)


def motordatavoyager4_get_datasets_for_eval_with_anomaly_label(data,
                                                               load_train=True,
                                                               load_test=True):
    """"
    Returns Sample Voyager4 Dataset For Evaluation Mode:

    Trainset samples include 80% normal samples and
    Testset includes normals (remaining 20%) + anomalies

    Label is anomaly status: 1: anomlay, 0:normal
    """

    eval_mode = True  # Test set includes validation normals
    label_as_signal = False

    speed_and_load_available = False

    return motordatavoyager4_get_datasets(data, load_train, load_test,
                                          eval_mode=eval_mode,
                                          label_as_signal=label_as_signal,
                                          speed_and_load_available=speed_and_load_available)


def motordatavoyager4_get_datasets_for_eval_with_signal(data,
                                                        load_train=True,
                                                        load_test=True):
    """"
    Returns Sample Voyager4 Dataset For Evaluation Mode:

    Trainset samples include 80% normal samples and
    Testset includes normals (remaining 20%) + anomalies

    Label is input signal
    """

    eval_mode = True
    label_as_signal = True

    speed_and_load_available = False

    return motordatavoyager4_get_datasets(data, load_train, load_test,
                                          eval_mode=eval_mode,
                                          label_as_signal=label_as_signal,
                                          speed_and_load_available=speed_and_load_available)


datasets = [
    {
        'name': 'MotorDataVoyager4_ForTrain',
        'input': (256, 3),
        'output': ('signal'),
        'regression': True,
        'loader': motordatavoyager4_get_datasets_for_train,
    },
    {
        'name': 'MotorDataVoyager4_ForEvalWithAnomalyLabel',
        'input': (256, 3),
        'output': ('normal', 'anomaly'),
        'loader': motordatavoyager4_get_datasets_for_eval_with_anomaly_label,
    },
    {
        'name': 'MotorDataVoyager4_ForEvalWithSignal',
        'input': (256, 3),
        'output': ('signal'),
        'loader': motordatavoyager4_get_datasets_for_eval_with_signal,
    }
]
