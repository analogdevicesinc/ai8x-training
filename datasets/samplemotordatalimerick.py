###################################################################################################
#
# Copyright (C) 2024 Analog Devices, Inc. All Rights Reserved.
# This software is proprietary to Analog Devices, Inc. and its licensors.
#
###################################################################################################
"""
Classes and functions for the Sample Motor Data Limerick Dataset
https://github.com/analogdevicesinc/CbM-Datasets
"""
import os

import numpy as np
import torch
from torchvision import transforms

import git
import pandas as pd
from git.exc import GitCommandError

import ai8x
from utils.dataloader_utils import makedir_exist_ok

from .cbm_dataframe_parser import CbM_DataFrame_Parser


class SampleMotorDataLimerick(CbM_DataFrame_Parser):
    """
    Sample motor data is collected using SpectraQuest Machinery Fault Simulator.
    ADXL356 sensor data is used for vibration raw data.
    For ADXL356 sensor, the sampling frequency was 20kHz and
    data csv files recorded for 2 sec in X, Y and Z direction.
    """

    # Good Bearing, Good Shaft, Balanced Load and Well Aligned
    healthy_file_identifier = '_GoB_GS_BaLo_WA_'

    num_end_zeros = 10
    num_start_zeros = 3

    train_ratio = 0.8

    def __init__(self, root, d_type,
                 transform,
                 target_sampling_rate_Hz,
                 signal_duration_in_sec,
                 overlap_ratio,
                 eval_mode,
                 label_as_signal,
                 random_or_speed_split,
                 speed_and_load_available,
                 num_end_zeros=num_end_zeros,
                 num_start_zeros=num_start_zeros,
                 train_ratio=train_ratio,
                 accel_in_second_dim=True,
                 download=True,
                 healthy_file_identifier=healthy_file_identifier,
                 cnn_1dinput_len=256):

        self.download = download
        self.root = root

        if self.download:
            self.__download()

        self.accel_in_second_dim = accel_in_second_dim

        self.processed_folder = \
            os.path.join(root, self.__class__.__name__, 'processed')

        self.healthy_file_identifier = healthy_file_identifier
        self.target_sampling_rate_Hz = target_sampling_rate_Hz
        self.signal_duration_in_sec = signal_duration_in_sec
        main_df = self.gen_dataframe()

        super().__init__(root,
                         d_type=d_type,
                         transform=transform,
                         target_sampling_rate_Hz=target_sampling_rate_Hz,
                         signal_duration_in_sec=signal_duration_in_sec,
                         overlap_ratio=overlap_ratio,
                         eval_mode=eval_mode,
                         label_as_signal=label_as_signal,
                         random_or_speed_split=random_or_speed_split,
                         speed_and_load_available=speed_and_load_available,
                         num_end_zeros=num_end_zeros,
                         num_start_zeros=num_start_zeros,
                         train_ratio=train_ratio,
                         cnn_1dinput_len=cnn_1dinput_len,
                         main_df=main_df)

    def __download(self):
        """
        Downloads Sample Motor Data Limerick dataset from:
        https://github.com/analogdevicesinc/CbM-Datasets
        """
        destination_folder = self.root
        dataset_repository = 'https://github.com/analogdevicesinc/CbM-Datasets'

        makedir_exist_ok(destination_folder)

        try:
            if not os.path.exists(os.path.join(destination_folder, 'SampleMotorDataLimerick')):
                print('\nDownloading SampleMotorDataLimerick dataset from\n'
                      f'{dataset_repository}\n')
                git.Repo.clone_from(dataset_repository, destination_folder)

            else:
                print('\nSampleMotorDataLimerick dataset already downloaded...')

        except GitCommandError:
            pass

    def parse_ADXL356C_and_return_common_df_row(self, file_full_path, sensor_sr_Hz,
                                                speed=None, load=None, label=None):
        """
        Dataframe parser for Sample Motor Data Limerick.
        Reads csv files and returns file identifier, raw data,
        sensor frequency, speed, load and label.
        The aw data size must be consecutive and bigger than window size.
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

        window_size_assert_message = "CNN input length is incorrect."
        assert self.signal_duration_in_sec <= (raw_data.shape[0] /
                                               sensor_sr_Hz), window_size_assert_message

        return [os.path.basename(file_full_path).split('/')[-1],
                raw_data, sensor_sr_Hz, speed, load, label]

    def __getitem__(self, index):
        if self.accel_in_second_dim and not self.speed_and_load_available:
            signal, lbl = super().__getitem__(index)  # pylint: disable=unbalanced-tuple-unpacking
            signal = torch.transpose(signal, 0, 1)
            lbl = lbl.transpose()
            return signal, lbl
        if self.accel_in_second_dim and self.speed_and_load_available:
            signal, lbl, speed, load = super().__getitem__(index)
            signal = torch.transpose(signal, 0, 1)
            lbl = lbl.transpose()
            return signal, lbl, speed, load
        return super().__getitem__(index)

    def gen_dataframe(self):
        """
        Generate dataframes from csv files of Sample Motor Data Limerick
        """
        file_name = f'{self.__class__.__name__}_dataframe.pkl'
        df_path = \
            os.path.join(self.root, self.__class__.__name__, file_name)

        if os.path.isfile(df_path):
            print(f'\nFile {file_name} already exists\n')
            main_df = pd.read_pickle(df_path)

            return main_df

        print('\nGenerating data frame pickle files from the raw data \n')

        actual_root_dir = os.path.join(self.root, self.__class__.__name__,
                                       "SpectraQuest_Rig_Data_Voyager_3/")
        data_dir = os.path.join(actual_root_dir, 'Data_ADXL356C')

        if not os.path.isdir(data_dir):
            print(f'\nDataset directory {data_dir} does not exist.\n')
            return None

        with os.scandir(data_dir) as it:
            if not any(it):
                print(f'\nDataset directory {data_dir} is empty.\n')
                return None

        rpm_prefixes = ('0600', '1200', '1800', '2400', '3000')

        sensor_sr_Hz = 20000  # Hz

        faulty_data_list = []
        healthy_data_list = []

        df_normals = self.create_common_empty_df()
        df_anormals = self.create_common_empty_df()

        for file in sorted(os.listdir(data_dir)):
            full_path = os.path.join(data_dir, file)
            speed = int(file.split("_")[0]) / 60  # Hz
            load = int(file.split("_")[-1][0:2])  # LBS

            if any(file.startswith(rpm_prefix + self.healthy_file_identifier)
                   for rpm_prefix in rpm_prefixes):
                healthy_row = self.parse_ADXL356C_and_return_common_df_row(
                    file_full_path=full_path, sensor_sr_Hz=sensor_sr_Hz,
                    speed=speed,
                    load=load,
                    label=0
                    )
                healthy_data_list.append(healthy_row)

            else:
                faulty_row = self.parse_ADXL356C_and_return_common_df_row(
                    file_full_path=full_path, sensor_sr_Hz=sensor_sr_Hz,
                    speed=speed,
                    load=load,
                    label=1
                    )
                faulty_data_list.append(faulty_row)

        df_normals = pd.DataFrame(data=np.array(healthy_data_list, dtype=object),
                                  columns=self.common_dataframe_columns)

        df_anormals = pd.DataFrame(data=np.array(faulty_data_list, dtype=object),
                                   columns=self.common_dataframe_columns)

        main_df = pd.concat([df_normals, df_anormals], axis=0)

        makedir_exist_ok(self.processed_folder)
        main_df.to_pickle(df_path)

        return main_df


def samplemotordatalimerick_get_datasets(data, load_train=True, load_test=True,
                                         download=True,
                                         signal_duration_in_sec=0.25,
                                         overlap_ratio=0.75,
                                         eval_mode=False,
                                         label_as_signal=True,
                                         random_or_speed_split=True,
                                         speed_and_load_available=False,
                                         accel_in_second_dim=True,
                                         target_sampling_rate_Hz=2000,
                                         cnn_1dinput_len=256):
    """
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
                                                signal_duration_in_sec=signal_duration_in_sec,
                                                overlap_ratio=overlap_ratio,
                                                eval_mode=eval_mode,
                                                label_as_signal=label_as_signal,
                                                random_or_speed_split=random_or_speed_split,
                                                speed_and_load_available=speed_and_load_available,
                                                accel_in_second_dim=accel_in_second_dim,
                                                target_sampling_rate_Hz=target_sampling_rate_Hz,
                                                cnn_1dinput_len=cnn_1dinput_len)

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
                                               signal_duration_in_sec=signal_duration_in_sec,
                                               overlap_ratio=overlap_ratio,
                                               eval_mode=eval_mode,
                                               label_as_signal=label_as_signal,
                                               random_or_speed_split=random_or_speed_split,
                                               speed_and_load_available=speed_and_load_available,
                                               accel_in_second_dim=accel_in_second_dim,
                                               target_sampling_rate_Hz=target_sampling_rate_Hz,
                                               cnn_1dinput_len=cnn_1dinput_len)

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

    eval_mode = False  # Test set includes validation normals
    label_as_signal = True

    signal_duration_in_sec = 0.25
    overlap_ratio = 0.75

    target_sampling_rate_Hz = 2000
    cnn_1dinput_len = 256

    # ds_ratio = 10,  sr: 20K / 10 = 2000, 0.25 sec window, fft input will have: 500 samples,
    # fftout's first 256 samples will be used
    # cnn input will have 256 samples

    accel_in_second_dim = True

    random_or_speed_split = True
    speed_and_load_available = False

    return samplemotordatalimerick_get_datasets(data, load_train, load_test,
                                                signal_duration_in_sec=signal_duration_in_sec,
                                                overlap_ratio=overlap_ratio,
                                                eval_mode=eval_mode,
                                                label_as_signal=label_as_signal,
                                                random_or_speed_split=random_or_speed_split,
                                                speed_and_load_available=speed_and_load_available,
                                                accel_in_second_dim=accel_in_second_dim,
                                                target_sampling_rate_Hz=target_sampling_rate_Hz,
                                                cnn_1dinput_len=cnn_1dinput_len)


def samplemotordatalimerick_get_datasets_for_eval_with_anomaly_label(data,
                                                                     load_train=True,
                                                                     load_test=True):
    """"
    Returns Sample Motor Data Limerick Dataset For Evaluation Mode
    Label is anomaly status
    """

    eval_mode = True  # Test set includes validation normals
    label_as_signal = False

    signal_duration_in_sec = 0.25
    overlap_ratio = 0.75

    target_sampling_rate_Hz = 2000
    cnn_1dinput_len = 256

    # ds_ratio = 10,  sr: 20K / 10 = 2000, 0.25 sec window, fft input will have: 500 samples,
    # fftout's first 256 samples will be used
    # cnn input will have 256 samples

    accel_in_second_dim = True

    random_or_speed_split = True
    speed_and_load_available = False

    return samplemotordatalimerick_get_datasets(data, load_train, load_test,
                                                signal_duration_in_sec=signal_duration_in_sec,
                                                overlap_ratio=overlap_ratio,
                                                eval_mode=eval_mode,
                                                label_as_signal=label_as_signal,
                                                random_or_speed_split=random_or_speed_split,
                                                speed_and_load_available=speed_and_load_available,
                                                accel_in_second_dim=accel_in_second_dim,
                                                target_sampling_rate_Hz=target_sampling_rate_Hz,
                                                cnn_1dinput_len=cnn_1dinput_len)


def samplemotordatalimerick_get_datasets_for_eval_with_signal(data,
                                                              load_train=True,
                                                              load_test=True):
    """"
    Returns Sample Motor Data Limerick Dataset For Evaluation Mode
    Label is signal
    """

    eval_mode = True  # Test set includes anormal samples as well as validation normals
    label_as_signal = True

    signal_duration_in_sec = 0.25
    overlap_ratio = 0.75

    target_sampling_rate_Hz = 2000
    cnn_1dinput_len = 256

    # ds_ratio = 10,  sr: 20K / 10 = 2000, 0.25 sec window, fft input will have: 500 samples,
    # fftout's first 256 samples will be used
    # cnn input will have 256 samples

    accel_in_second_dim = True

    random_or_speed_split = True
    speed_and_load_available = False

    return samplemotordatalimerick_get_datasets(data, load_train, load_test,
                                                signal_duration_in_sec=signal_duration_in_sec,
                                                overlap_ratio=overlap_ratio,
                                                eval_mode=eval_mode,
                                                label_as_signal=label_as_signal,
                                                random_or_speed_split=random_or_speed_split,
                                                speed_and_load_available=speed_and_load_available,
                                                accel_in_second_dim=accel_in_second_dim,
                                                target_sampling_rate_Hz=target_sampling_rate_Hz,
                                                cnn_1dinput_len=cnn_1dinput_len)


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
