###################################################################################################
#
# Copyright (C) 2024 Analog Devices, Inc. All Rights Reserved.
# This software is proprietary to Analog Devices, Inc. and its licensors.
#
###################################################################################################
"""
Classes and functions for the Limerick Sample Motor Data Dataset
"""
import errno
import math
import os
import pickle

import numpy as np
import torch
from numpy.fft import fft
from torch.utils.data import Dataset
from torchvision import transforms

import git
import pandas as pd
import scipy
from git.exc import GitCommandError

import ai8x


class SampleMotorDataLimerick(Dataset):
    """
    Sample motor data is collected using SpectraQuest Machinery Fault Simulator.
    ADXL356 sensor data is used for vibration raw data.
    For ADXL356 sensor, the sampling frequency was 20kHz and
    data csv files recorded for 2 sec in X, Y and Z direction.
    """

    # Order 0 is reserved for 'all' do not change order
    rpm_options = ('all', '0600', '1200', '1800', '2400', '3000')
    sensor_sr_Hz = 20000

    # Good Bearing, Good Shaft, Balanced Load and Well Aligned
    healthy_file_identifier = '_GoB_GS_BaLo_WA_'

    cnn_1dinput_len = 256

    num_end_zeros = 10
    num_start_zeros = 3

    common_dataframe_columns = ["sensor_identifier", "file_identifier", "raw_data_accel_in_g"]

    train_ratio = 0.8

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
            result_list[i, :, :] = SampleMotorDataLimerick.sliding_windows_1d(
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

        sliding_windows = SampleMotorDataLimerick.sliding_windows_on_columns_of_2d(
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

        file_raw_data_windows = SampleMotorDataLimerick.split_file_raw_data(
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

        fft_output_window_size = SampleMotorDataLimerick.cnn_1dinput_len

        file_cnn_signals = np.zeros((num_features, num_windows, fft_output_window_size))

        # Perform FFT on each window () for each feature
        for window in range(num_windows):
            for feature in range(num_features):

                signal_for_fft = file_raw_data_windows[feature, window, :]

                fft_out = abs(fft(signal_for_fft))
                fft_out = fft_out[:fft_output_window_size]

                fft_out[:SampleMotorDataLimerick.num_start_zeros] = 0
                fft_out[-SampleMotorDataLimerick.num_end_zeros:] = 0

                file_cnn_signals[feature, window, :] = fft_out

            file_cnn_signals[:, window, :] = file_cnn_signals[:, window, :] / \
                np.sqrt(np.power(file_cnn_signals[:, window, :], 2).sum())

        # Reshape from (num_features, num_windows, window_size) into:
        # (num_windows, num_features, window_size)
        file_cnn_signals = file_cnn_signals.transpose([1, 0, 2])

        return file_cnn_signals

    @staticmethod
    def create_common_empty_df():
        """
        Create empty dataframe
        """
        df = pd.DataFrame(columns=SampleMotorDataLimerick.common_dataframe_columns)
        return df

    @staticmethod
    def parse_ADXL356C_and_return_common_df_row(file_full_path):
        """
        Columns added just for readability can return raw data np array as well,
        can also add file identifier
        """
        df_raw = pd.read_csv(file_full_path, sep=';', header=None)
        df_raw.rename(
            columns={0: 'Time', 1: 'Voltage_x', 2: 'Voltage_y',
                     3: 'Voltage_z', 4: 'x', 5: 'y', 6: 'z'},
            inplace=True
            )
        ss_vibr_x1 = df_raw.iloc[0]['x']
        ss_vibr_y1 = df_raw.iloc[0]['y']
        ss_vibr_z1 = df_raw.iloc[0]['z']
        df_raw["Acceleration_x (g)"] = 50 * (df_raw["Voltage_x"] - ss_vibr_x1)
        df_raw["Acceleration_y (g)"] = 50 * (df_raw["Voltage_y"] - ss_vibr_y1)
        df_raw["Acceleration_z (g)"] = 50 * (df_raw["Voltage_z"] - ss_vibr_z1)

        raw_data = df_raw[["Acceleration_x (g)", "Acceleration_y (g)", "Acceleration_z (g)"]]
        raw_data = raw_data.to_numpy()
        return ['ADXL356C', os.path.basename(file_full_path).split('/')[-1], raw_data]

    def __makedir_exist_ok(self, dirpath):
        try:
            os.makedirs(dirpath)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

    def __init__(self, root, d_type, transform=None,
                 download=True,
                 downsampling_ratio=2,
                 signal_duration_in_sec=0.25,
                 overlap_ratio=0.75,
                 eval_mode=False,
                 label_as_signal=True,
                 random_or_speed_split=True,
                 accel_in_second_dim=True,
                 sensor_selected='ADXL356C',
                 rpm_selected=rpm_options[0]):

        if d_type not in ('test', 'train'):
            raise ValueError(
                "d_type can only be set to 'test' or 'train'"
                )

        if rpm_selected not in SampleMotorDataLimerick.rpm_options:
            raise ValueError(
                f"rpm_selected can only be set from: {SampleMotorDataLimerick.rpm_options}"
                )

        if not isinstance(downsampling_ratio, int) or downsampling_ratio < 1:
            raise ValueError(
                "downsampling_ratio can only be set to an integer value greater than 0"
                )

        self.selected_sensor_sr = SampleMotorDataLimerick.sensor_sr_Hz

        self.root = root
        self.d_type = d_type
        self.transform = transform
        self.download = download

        self.downsampling_ratio = downsampling_ratio
        self.signal_duration_in_sec = signal_duration_in_sec
        self.overlap_ratio = overlap_ratio

        self.eval_mode = eval_mode
        self.label_as_signal = label_as_signal

        self.random_or_speed_split = random_or_speed_split
        self.accel_in_second_dim = accel_in_second_dim

        self.sensor_selected = sensor_selected
        self.rpm_selected = rpm_selected

        self.num_of_features = 3

        if self.download:
            self.__download()

        processed_folder = \
            os.path.join(root, self.__class__.__name__, 'processed')

        self.__makedir_exist_ok(processed_folder)

        self.specs_identifier = f'eval_mode_{self.eval_mode}_' + \
                                f'label_as_signal_{self.label_as_signal}_' + \
                                f'ds_{self.downsampling_ratio}_' + \
                                f'dur_{self.signal_duration_in_sec}_' + \
                                f'ovlp_ratio_{self.overlap_ratio}_' + \
                                f'random_split_{self.random_or_speed_split}_' + \
                                f'rpm_{self.rpm_selected}'

        train_dataset_pkl_file_path = \
            os.path.join(processed_folder, f'train_{self.specs_identifier}.pkl')

        test_dataset_pkl_file_path =  \
            os.path.join(processed_folder, f'test_{self.specs_identifier}.pkl')

        if self.d_type == 'train':
            self.dataset_pkl_file_path = train_dataset_pkl_file_path

        elif self.d_type == 'test':
            self.dataset_pkl_file_path = test_dataset_pkl_file_path

        self.signal_list = []
        self.lbl_list = []

        self.__create_pkl_files()
        self.is_truncated = False

    def __download(self):
        """
        Downloads Sample Motor Data Limerick dataset from:
        https://github.com/analogdevicesinc/CbM-Datasets
        """
        destination_folder = self.root
        dataset_repository = 'https://github.com/analogdevicesinc/CbM-Datasets'

        self.__makedir_exist_ok(destination_folder)

        try:
            if not os.path.exists(os.path.join(destination_folder, 'SampleMotorDataLimerick')):
                print('\nDownloading SampleMotorDataLimerick dataset from\n'
                      f'{dataset_repository}\n')
                git.Repo.clone_from(dataset_repository, destination_folder)

            else:
                print('\nSampleMotorDataLimerick dataset already downloaded...')

        except GitCommandError:
            pass

    def __create_pkl_files(self):
        if os.path.exists(self.dataset_pkl_file_path):

            print('\nPickle files are already generated ...\n')

            (self.signal_list, self.lbl_list) = \
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
        print('\nGenerating dataset pickle files from the raw data '
              f'files (specs identifier: {self.specs_identifier}) ...\n')

        actual_root_dir = os.path.join(self.root, self.__class__.__name__,
                                       "SpectraQuest_Rig_Data_Voyager_3/")

        data_dir = os.path.join(actual_root_dir, f'Data_{self.sensor_selected}/')

        if not os.listdir(data_dir):
            print('\nDataset directory is empty.\n')

        selected_rpm_prefixes = (
            SampleMotorDataLimerick.rpm_options[1:]
            if self.rpm_selected == SampleMotorDataLimerick.rpm_options[0] else self.rpm_selected
            )

        faulty_data_list = []
        healthy_data_list = []

        df_normals = SampleMotorDataLimerick.create_common_empty_df()
        df_anormals = SampleMotorDataLimerick.create_common_empty_df()

        for file in os.listdir(data_dir):
            full_path = os.path.join(data_dir, file)

            if any(file.startswith(rpm_prefix + SampleMotorDataLimerick.healthy_file_identifier)
                   for rpm_prefix in selected_rpm_prefixes):
                if self.sensor_selected == 'ADXL356C':
                    healthy_row = SampleMotorDataLimerick.parse_ADXL356C_and_return_common_df_row(
                        file_full_path=full_path
                        )
                healthy_data_list.append(healthy_row)

            else:
                if self.sensor_selected == 'ADXL356C':
                    faulty_row = SampleMotorDataLimerick.parse_ADXL356C_and_return_common_df_row(
                        file_full_path=full_path
                        )

                faulty_data_list.append(faulty_row)

        # Can keep and process those further
        df_normals = pd.DataFrame(data=np.array(healthy_data_list, dtype=object),
                                  columns=SampleMotorDataLimerick.common_dataframe_columns)
        df_anormals = pd.DataFrame(data=np.array(faulty_data_list, dtype=object),
                                   columns=SampleMotorDataLimerick.common_dataframe_columns)

        # LOAD NORMAL FEATURES
        test_train_idx_max = 4
        test_train_idx = 0  # 0, 1, 2 : train, 3: test

        train_features = []
        test_normal_features = []

        for _, row in df_normals.iterrows():
            raw_data = row['raw_data_accel_in_g']
            cnn_signals = self.process_file_and_return_signal_windows(raw_data)
            if self.random_or_speed_split:
                num_training = int(SampleMotorDataLimerick.train_ratio * cnn_signals.shape[0])

                for i in range(cnn_signals.shape[0]):
                    if i < num_training:
                        train_features.append(cnn_signals[i])
                    else:
                        test_normal_features.append(cnn_signals[i])
            else:
                if test_train_idx < test_train_idx_max - 1:
                    for i in range(cnn_signals.shape[0]):
                        train_features.append(cnn_signals[i])
                else:
                    for i in range(cnn_signals.shape[0]):
                        test_normal_features.append(cnn_signals[i])

                test_train_idx = (test_train_idx + 1) % test_train_idx_max

        train_features = np.asarray(train_features)
        test_normal_features = np.asarray(test_normal_features)

        anomaly_features = []

        for _, row in df_anormals.iterrows():
            raw_data = row['raw_data_accel_in_g']
            cnn_signals = self.process_file_and_return_signal_windows(raw_data)
            for i in range(cnn_signals.shape[0]):
                anomaly_features.append(cnn_signals[i])

        anomaly_features = np.asarray(anomaly_features)

        train_features = self.normalize_signal(train_features)
        test_normal_features = self.normalize_signal(test_normal_features)
        anomaly_features = self.normalize_signal(anomaly_features)

        # For eliminating filter effects
        train_features[:, :, :SampleMotorDataLimerick.num_start_zeros] = 0.5
        train_features[:, :, -SampleMotorDataLimerick.num_end_zeros:] = 0.5

        test_normal_features[:, :, :SampleMotorDataLimerick.num_start_zeros] = 0.5
        test_normal_features[:, :, -SampleMotorDataLimerick.num_end_zeros:] = 0.5

        anomaly_features[:, :, :SampleMotorDataLimerick.num_start_zeros] = 0.5
        anomaly_features[:, :, -SampleMotorDataLimerick.num_end_zeros:] = 0.5

        # ARRANGE TEST-TRAIN SPLIT AND LABELS
        if self.d_type == 'train':
            self.lbl_list = [train_features[i, :, :] for i in range(train_features.shape[0])]
            self.signal_list = [torch.Tensor(label) for label in self.lbl_list]
            self.lbl_list = list(self.signal_list)

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

            if not self.label_as_signal:
                self.lbl_list = np.concatenate(
                                    (np.zeros([len(test_normal_features), 1]),
                                     np.ones([len(anomaly_features), 1])), axis=0)
        # Save pickle file
        pickle.dump((self.signal_list, self.lbl_list), open(self.dataset_pkl_file_path, 'wb'))

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

        if self.accel_in_second_dim:
            signal = torch.transpose(signal, 0, 1)
            lbl = lbl.transpose()

        return signal, lbl


def samplemotordatalimerick_get_datasets(data, load_train=True, load_test=True,
                                         download=True,
                                         downsampling_ratio=10,
                                         signal_duration_in_sec=0.25,
                                         overlap_ratio=0.75,
                                         eval_mode=False,
                                         label_as_signal=True,
                                         random_or_speed_split=True,
                                         accel_in_second_dim=True,
                                         sensor_selected='ADXL356C',
                                         rpm_selected=SampleMotorDataLimerick.rpm_options[0]):
    """"
    Returns Sample Motor Data Limerick Dataset
    """
    (data_dir, args) = data

    if load_train:
        train_transform = transforms.Compose([
            ai8x.normalize(args=args)
        ])

        train_dataset = SampleMotorDataLimerick(root=data_dir, d_type='train',
                                                download=download,
                                                transform=train_transform,
                                                downsampling_ratio=downsampling_ratio,
                                                signal_duration_in_sec=signal_duration_in_sec,
                                                overlap_ratio=overlap_ratio,
                                                eval_mode=eval_mode,
                                                label_as_signal=label_as_signal,
                                                random_or_speed_split=random_or_speed_split,
                                                accel_in_second_dim=accel_in_second_dim,
                                                sensor_selected=sensor_selected,
                                                rpm_selected=rpm_selected)

        print(f'Train dataset length: {len(train_dataset)}\n')
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            ai8x.normalize(args=args)
        ])

        test_dataset = SampleMotorDataLimerick(root=data_dir, d_type='test',
                                               download=download,
                                               transform=test_transform,
                                               downsampling_ratio=downsampling_ratio,
                                               signal_duration_in_sec=signal_duration_in_sec,
                                               overlap_ratio=overlap_ratio,
                                               eval_mode=eval_mode,
                                               label_as_signal=label_as_signal,
                                               random_or_speed_split=random_or_speed_split,
                                               accel_in_second_dim=accel_in_second_dim,
                                               sensor_selected=sensor_selected,
                                               rpm_selected=rpm_selected)

        print(f'Test dataset length: {len(test_dataset)}\n')
    else:
        test_dataset = None

    return train_dataset, test_dataset


def samplemotordatalimerick_get_datasets_for_train(data,
                                                   load_train=True,
                                                   load_test=True):
    """"
    Returns Sample Motor Data Limerick Dataset For Training Mode
    """

    eval_mode = False   # Test set includes validation normals
    label_as_signal = True

    signal_duration_in_sec = 0.25
    overlap_ratio = 0.75

    wanted_sampling_rate_Hz = 2000
    downsampling_ratio = round(SampleMotorDataLimerick.sensor_sr_Hz /
                               wanted_sampling_rate_Hz)

    # ds_ratio = 10,  sr: 20K / 10 = 2000, 0.25 sec window, fft input will have: 500 samples,
    # fftout's first 256 samples will be used
    # cnn input will have 256 samples

    accel_in_second_dim = True

    random_or_speed_split = True

    return samplemotordatalimerick_get_datasets(data, load_train, load_test,
                                                downsampling_ratio=downsampling_ratio,
                                                signal_duration_in_sec=signal_duration_in_sec,
                                                overlap_ratio=overlap_ratio,
                                                eval_mode=eval_mode,
                                                label_as_signal=label_as_signal,
                                                random_or_speed_split=random_or_speed_split,
                                                accel_in_second_dim=accel_in_second_dim)


def samplemotordatalimerick_get_datasets_for_eval_with_anomaly_label(data,
                                                                     load_train=True,
                                                                     load_test=True):
    """"
    Returns Sample Motor Data Limerick Dataset For Evaluation Mode
    Label is anomaly status
    """

    eval_mode = True   # Test set includes validation normals
    label_as_signal = False

    signal_duration_in_sec = 0.25
    overlap_ratio = 0.75

    wanted_sampling_rate_Hz = 2000
    downsampling_ratio = round(SampleMotorDataLimerick.sensor_sr_Hz /
                               wanted_sampling_rate_Hz)

    # ds_ratio = 10,  sr: 20K / 10 = 2000, 0.25 sec window, fft input will have: 500 samples,
    # fftout's first 256 samples will be used
    # cnn input will have 2556 samples

    accel_in_second_dim = True

    random_or_speed_split = True

    return samplemotordatalimerick_get_datasets(data, load_train, load_test,
                                                downsampling_ratio=downsampling_ratio,
                                                signal_duration_in_sec=signal_duration_in_sec,
                                                overlap_ratio=overlap_ratio,
                                                eval_mode=eval_mode,
                                                label_as_signal=label_as_signal,
                                                random_or_speed_split=random_or_speed_split,
                                                accel_in_second_dim=accel_in_second_dim)


def samplemotordatalimerick_get_datasets_for_eval_with_signal(data,
                                                              load_train=True,
                                                              load_test=True):
    """"
    Returns Sample Motor Data Limerick Dataset For Evaluation Mode
    Label is signal
    """

    eval_mode = True   # Test set includes validation normals
    label_as_signal = True

    signal_duration_in_sec = 0.25
    overlap_ratio = 0.75

    wanted_sampling_rate_Hz = 2000
    downsampling_ratio = round(SampleMotorDataLimerick.sensor_sr_Hz /
                               wanted_sampling_rate_Hz)

    # ds_ratio = 10,  sr: 20K / 10 = 2000, 0.25 sec window, fft input will have: 500 samples,
    # fftout's first 256 samples will be used
    # cnn input will have 2556 samples

    accel_in_second_dim = True

    random_or_speed_split = True

    return samplemotordatalimerick_get_datasets(data, load_train, load_test,
                                                downsampling_ratio=downsampling_ratio,
                                                signal_duration_in_sec=signal_duration_in_sec,
                                                overlap_ratio=overlap_ratio,
                                                eval_mode=eval_mode,
                                                label_as_signal=label_as_signal,
                                                random_or_speed_split=random_or_speed_split,
                                                accel_in_second_dim=accel_in_second_dim)


datasets = [
    {
        'name': 'SampleMotorDataLimerick_ForTrain',
        'input': (256, 3),
        'output': ('signal'),
        'regression': True,
        'loader': samplemotordatalimerick_get_datasets_for_train,
    },
    {
        'name': 'SampleMotorDataLimerick_ForEvalWithAnomalyLabel',
        'input': (256, 3),
        'output': ('normal', 'anomaly'),
        'loader': samplemotordatalimerick_get_datasets_for_eval_with_anomaly_label,
    },
    {
        'name': 'SampleMotorDataLimerick_ForEvalWithSignal',
        'input': (256, 3),
        'output': ('signal'),
        'loader': samplemotordatalimerick_get_datasets_for_eval_with_signal,
    }
]
