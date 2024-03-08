###################################################################################################
#
# Copyright (C) 2024 Analog Devices, Inc. All Rights Reserved.
# This software is proprietary to Analog Devices, Inc. and its licensors.
#
###################################################################################################
"""
Main classes and functions for Motor Data Dataset
"""
import math
import os
import pickle

import numpy as np
import torch
from numpy.fft import fft
from torch.utils.data import Dataset

import pandas as pd
import scipy

from utils.dataloader_utils import makedir_exist_ok


class CbM_DataFrame_Parser(Dataset):  # pylint: disable=too-many-instance-attributes
    """
    The base dataset class for motor vibration data used in Condition Based Monitoring.
    Includes main preprocessing functions.
    Expects a dataframe with common_dataframe_columns.
    """

    common_dataframe_columns = ["file_identifier", "raw_data_vib_in_g", "sensor_sr_Hz",
                                "speed", "load", "label"]

    @staticmethod
    def sliding_windows_1d(array, window_size, overlap_ratio):
        """
        One dimensional array is windowed and returned
        in window_size length according to overlap ratio.
        """

        window_overlap = math.ceil(window_size * overlap_ratio)

        slide_amount = window_size - window_overlap
        num_of_windows = math.floor((len(array) - window_size) / slide_amount) + 1

        result_list = np.zeros((num_of_windows, window_size))

        for i in range(num_of_windows):
            start_idx = slide_amount * i
            end_idx = start_idx + window_size
            result_list[i] = array[start_idx:end_idx]

        return result_list

    @staticmethod
    def sliding_windows_on_columns_of_2d(array, window_size, overlap_ratio):
        """
        Two dimensional array is windowed and returned
        in window_size length according to overlap ratio.
        """

        array_len, num_of_cols = array.shape

        window_overlap = math.ceil(window_size * overlap_ratio)
        slide_amount = window_size - window_overlap
        num_of_windows = math.floor((array_len - window_size) / slide_amount) + 1

        result_list = np.zeros((num_of_cols, num_of_windows, window_size))

        for i in range(num_of_cols):
            result_list[i, :, :] = CbM_DataFrame_Parser.sliding_windows_1d(
                array[:, i],
                window_size, overlap_ratio
                )

        return result_list

    @staticmethod
    def split_file_raw_data(file_raw_data, file_raw_data_fs_in_Hz, duration_in_sec, overlap_ratio):
        """
        Raw data is split into windowed data.
        """

        num_of_samples_per_window = int(file_raw_data_fs_in_Hz * duration_in_sec)

        sliding_windows = CbM_DataFrame_Parser.sliding_windows_on_columns_of_2d(
            file_raw_data,
            num_of_samples_per_window,
            overlap_ratio
            )

        return sliding_windows

    def process_file_and_return_signal_windows(self, file_raw_data):
        """
        Windowed signals are constructed from 2D raw data.
        Fast Fourier Transform performed on these signals.
        """

        new_sampling_rate = int(self.selected_sensor_sr / self.downsampling_ratio)

        file_raw_data_sampled = scipy.signal.decimate(file_raw_data,
                                                      self.downsampling_ratio, axis=0)

        file_raw_data_windows = self.split_file_raw_data(
            file_raw_data_sampled,
            new_sampling_rate,
            self.signal_duration_in_sec,
            self.overlap_ratio
            )

        # First dimension: 3
        # Second dimension: number of windows
        # Third dimension: Window for self.duration_in_sec. 1000 samples for default settings
        num_features = file_raw_data_windows.shape[0]
        num_windows = file_raw_data_windows.shape[1]

        fft_output_window_size = self.cnn_1dinput_len

        file_cnn_signals = np.zeros((num_features, num_windows, fft_output_window_size))

        # Perform FFT on each window () for each feature
        for window in range(num_windows):
            for feature in range(num_features):

                signal_for_fft = file_raw_data_windows[feature, window, :]

                fft_out = abs(fft(signal_for_fft))
                fft_out = fft_out[:fft_output_window_size]

                fft_out[:self.num_start_zeros] = 0
                fft_out[-self.num_end_zeros:] = 0

                file_cnn_signals[feature, window, :] = fft_out

            file_cnn_signals[:, window, :] = file_cnn_signals[:, window, :] / \
                np.sqrt(np.power(file_cnn_signals[:, window, :], 2).sum())

        # Reshape from (num_features, num_windows, window_size) into:
        # (num_windows, num_features, window_size)
        file_cnn_signals = file_cnn_signals.transpose([1, 0, 2])

        return file_cnn_signals

    def create_common_empty_df(self):
        """
        Create empty dataframe
        """
        df = pd.DataFrame(columns=self.common_dataframe_columns)
        return df

    def __init__(self, root, d_type,
                 transform=None,
                 target_sampling_rate_Hz=2000,
                 signal_duration_in_sec=0.25,
                 overlap_ratio=0.75,
                 eval_mode=False,
                 label_as_signal=True,
                 random_or_speed_split=True,
                 speed_and_load_available=False,
                 num_end_zeros=10,
                 num_start_zeros=3,
                 train_ratio=0.8,
                 cnn_1dinput_len=256,
                 main_df=None
                 ):

        if d_type not in ('test', 'train'):
            raise ValueError(
                "d_type can only be set to 'test' or 'train'"
                )

        self.main_df = main_df
        self.df_normals = self.main_df[main_df['label'] == 0]
        self.df_anormals = self.main_df[main_df['label'] == 1]

        self.normal_speeds_Hz = list(set(self.df_normals['speed']))
        self.normal_speeds_Hz.sort()
        self.normal_test_speeds = self.normal_speeds_Hz[1::5]
        self.normal_train_speeds = list(set(self.normal_speeds_Hz) - set(self.normal_test_speeds))
        self.normal_train_speeds.sort()

        self.selected_sensor_sr = self.df_normals['sensor_sr_Hz'][0]
        self.num_end_zeros = num_end_zeros
        self.num_start_zeros = num_start_zeros
        self.train_ratio = train_ratio

        self.root = root
        self.d_type = d_type
        self.transform = transform

        self.signal_duration_in_sec = signal_duration_in_sec
        self.overlap_ratio = overlap_ratio

        self.eval_mode = eval_mode
        self.label_as_signal = label_as_signal

        self.random_or_speed_split = random_or_speed_split
        self.speed_and_load_available = speed_and_load_available

        self.num_of_features = 3

        self.target_sampling_rate_Hz = target_sampling_rate_Hz
        self.downsampling_ratio = round(self.selected_sensor_sr /
                                        self.target_sampling_rate_Hz)

        self.cnn_1dinput_len = cnn_1dinput_len

        cnn_assert_message = "CNN input length is incorrect."
        assert self.cnn_1dinput_len >= (self.target_sampling_rate_Hz *
                                        self.signal_duration_in_sec)/2, cnn_assert_message

        if not isinstance(self.downsampling_ratio, int) or self.downsampling_ratio < 1:
            raise ValueError(
                "downsampling_ratio can only be set to an integer value greater than 0"
                )

        processed_folder = \
            os.path.join(root, self.__class__.__name__, 'processed')

        self.processed_folder = processed_folder

        makedir_exist_ok(self.processed_folder)

        self.specs_identifier = f'eval_mode_{self.eval_mode}_' + \
                                f'label_as_signal_{self.label_as_signal}_' + \
                                f'ds_{self.downsampling_ratio}_' + \
                                f'dur_{self.signal_duration_in_sec}_' + \
                                f'ovlp_ratio_{self.overlap_ratio}_' + \
                                f'random_split_{self.random_or_speed_split}_'

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

    def __create_pkl_files(self):
        if os.path.exists(self.dataset_pkl_file_path):

            print('\nPickle files are already generated ...\n')

            (self.signal_list, self.lbl_list, self.speed_list, self.load_list) = \
                pickle.load(open(self.dataset_pkl_file_path, 'rb'))
            return

        self.__gen_datasets()

    def normalize_signal(self, features):
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

    def __gen_datasets(self):

        train_features = []
        test_normal_features = []

        train_speeds = []
        test_normal_speeds = []

        train_loads = []
        test_normal_loads = []

        for _, row in self.df_normals.iterrows():
            raw_data = row['raw_data_vib_in_g']
            cnn_signals = self.process_file_and_return_signal_windows(raw_data)
            file_speed = row['speed']
            file_load = row['load']

            if self.random_or_speed_split:
                num_training = int(self.train_ratio * cnn_signals.shape[0])

                for i in range(cnn_signals.shape[0]):
                    if i < num_training:
                        train_features.append(cnn_signals[i])
                        train_speeds.append(file_speed)
                        train_loads.append(file_load)
                    else:
                        test_normal_features.append(cnn_signals[i])
                        test_normal_speeds.append(file_speed)
                        test_normal_loads.append(file_load)

            else:
                # split test-train using file identifiers and split
                if file_speed in self.normal_train_speeds:
                    for i in range(cnn_signals.shape[0]):
                        train_features.append(cnn_signals[i])
                        train_speeds.append(file_speed)
                        train_loads.append(file_load)

                else:  # file_speed in normal_test_speeds
                    for i in range(cnn_signals.shape[0]):
                        test_normal_features.append(cnn_signals[i])
                        test_normal_speeds.append(file_speed)
                        test_normal_loads.append(file_load)

        train_features = np.asarray(train_features)
        test_normal_features = np.asarray(test_normal_features)

        anomaly_features = []
        test_anormal_speeds = []
        test_anormal_loads = []

        for _, row in self.df_anormals.iterrows():
            raw_data = row['raw_data_vib_in_g']
            cnn_signals = self.process_file_and_return_signal_windows(raw_data)
            file_speed = row['speed']
            file_load = row['load']

            for i in range(cnn_signals.shape[0]):
                anomaly_features.append(cnn_signals[i])
                test_anormal_speeds.append(file_speed)
                test_anormal_loads.append(file_load)

        anomaly_features = np.asarray(anomaly_features)

        train_features = self.normalize_signal(train_features)
        test_normal_features = self.normalize_signal(test_normal_features)
        anomaly_features = self.normalize_signal(anomaly_features)

        # For eliminating filter effects
        train_features[:, :, :self.num_start_zeros] = 0.5
        train_features[:, :, -self.num_end_zeros:] = 0.5

        test_normal_features[:, :, :self.num_start_zeros] = 0.5
        test_normal_features[:, :, -self.num_end_zeros:] = 0.5

        anomaly_features[:, :, :self.num_start_zeros] = 0.5
        anomaly_features[:, :, -self.num_end_zeros:] = 0.5

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
        lbl = self.lbl_list[index]

        if self.transform is not None:
            signal = self.transform(signal)

            if self.label_as_signal:
                lbl = self.transform(lbl)

        if not self.label_as_signal:
            lbl = lbl.astype(np.long)
        else:
            lbl = lbl.numpy().astype(np.float32)

        if self.speed_and_load_available:
            speed = self.speed_list[index]
            load = self.load_list[index]

            return signal, lbl, speed, load

        return signal, lbl
