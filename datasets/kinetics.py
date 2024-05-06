###################################################################################################
#
# Copyright (C) 2022-2023 Maxim Integrated Products, Inc. All Rights Reserved.
# Copyright (C) 2024 Analog Devices, Inc. All Rights Reserved.
#
# This software is proprietary to Analog Devices, Inc. and its licensors.
#
###################################################################################################
"""
Classes and functions used to utilize the Kinetics dataset.
"""
import errno
import os
import pickle
import tarfile
import urllib
import urllib.error
import urllib.request
from zipfile import ZipFile

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import albumentations as A
import cv2
import pandas as pd
import yaml
from pyffmpeg import FFmpeg
from pytube import YouTube  # type: ignore # class YouTube is in pytube.__main__
from pytube.exceptions import VideoUnavailable
from tqdm import tqdm

import ai8x


class Kinetics(Dataset):
    """
    Kinetics400 Human Actions Dataset (400 action class)
    (https://deepmind.com/research/open-source/kinetics/).
    """

    url_kinetics400 = 'https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz'

    def __init__(self, root, split, img_size, classes, num_frames_model,
                 num_frames_dataset, max_examples_per_class, transform, augmentation,
                 blacklist_file=None, download=True):

        self.root = root
        self.split = split
        self.img_size = img_size
        self.classes = classes
        self.num_frames_model = num_frames_model
        self.num_frames_dataset = num_frames_dataset
        self.max_examples_per_class = max_examples_per_class
        self.transform = transform
        self.augmentation = augmentation
        self.blacklist_file = blacklist_file
        self.dataset = []

        # Check split
        if split not in ('test', 'train'):
            raise ValueError("Split name can only be set to 'test' or 'train'")

        # Download dataset
        if download:
            self.__download()

        self.__load_dataset()

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

        self.__makedir_exist_ok(self.processed_folder)
        self.__makedir_exist_ok(os.path.join(self.processed_folder, self.split))
        if self.__check_processed_exists():
            return  # skip download if dataset already exists

        self.__makedir_exist_ok(self.raw_folder)
        if not self.__check_dataset_exists():
            # Download the dataset file containing the video links
            filename = self.url_kinetics400.rpartition('/')[2]
            self.__download_and_extract_archive(self.url_kinetics400,
                                                download_root=self.raw_folder,
                                                filename=filename)

        self.__download_videos()
        self.__pickle_videos()

    def __check_processed_exists(self):
        # check if the number of processed & pickled files matches with the number of classes
        num_processed_files = len([f for f in os.listdir(os.path.join(
            self.processed_folder, self.split)) if f.endswith('.pkl')])
        return num_processed_files == len(self.classes)

    def __check_dataset_exists(self):
        return os.path.exists(os.path.join(self.raw_folder, 'kinetics400', self.split + '.csv'))

    def __makedir_exist_ok(self, dirpath):
        try:
            os.makedirs(dirpath)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

    def __download_url(self, url, root, filename=None):
        root = os.path.expanduser(root)
        if not filename:
            filename = os.path.basename(url)
        fpath = os.path.join(root, filename)

        self.__makedir_exist_ok(root)

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
                                       remove_finished=False):
        download_root = os.path.expanduser(download_root)
        if extract_root is None:
            extract_root = download_root
        if not filename:
            filename = os.path.basename(url)

        self.__download_url(url, download_root, filename)

        archive = os.path.join(download_root, filename)
        print(f"Extracting {archive} to {extract_root}")
        self.__extract_archive(archive, extract_root, remove_finished)

    def __download_videos(self):

        youtube_download_link_prefix = "https://www.youtube.com/watch?v="
        csv_file = os.path.join(self.raw_folder, 'kinetics400/' + self.split + '.csv')
        df_set = pd.read_csv(csv_file)
        download_folder = os.path.join(self.raw_folder, self.split)
        downloads_per_class = np.zeros(len(self.classes),  dtype=int)
        for cls in self.classes:
            self.__makedir_exist_ok(os.path.join(download_folder, cls))

        df_set = df_set.reset_index()  # make sure indexes pair with number of rows

        vids_to_download = self.max_examples_per_class
        for class_name in self.classes[:-1]:
            vids_to_download += min((df_set.label == class_name).sum(),
                                    self.max_examples_per_class)

        print(f'The goal number of {self.split} set videos is {vids_to_download}')
        print('Note: Due to deleted videos in the dataset, the goal number may not be reached')
        print('Logging into a Youtube account for automated downloading may be necessary.',
              'Please follow any instructions to enter codes in browser if prompted')

        ff = FFmpeg()  # initialize ffmpeg object

        with tqdm(total=vids_to_download) as pbar:
            pbar.set_description(f'Downloading and Processing {self.split} set')
            for _, row in df_set.iterrows():
                if row.label in self.classes:
                    class_index = self.classes.index(row.label)
                else:
                    class_index = -1
                # download the video if the goal is not reached for the class
                if downloads_per_class[class_index] != self.max_examples_per_class:
                    video_base_name = row.split + '_' + f'{row.name:05.0f}'
                    youtube_download_link = youtube_download_link_prefix + row.youtube_id
                    is_downloaded = self.__download_and_crop_video(
                        ff, youtube_download_link,
                        os.path.join(download_folder, self.classes[class_index]),
                        video_base_name, row.time_start, row.time_end)
                    if is_downloaded:
                        downloads_per_class[class_index] += 1
                        pbar.update(1)

        print(f'The number of processed {self.split} set videos is {sum(downloads_per_class)}')

    def __download_and_crop_video(self, ff, youtube_download_link, save_folder,
                                  video_base_filename, t_beg, t_end):

        video_filename_proc = video_base_filename + '_proc.mp4'
        if os.path.exists(os.path.join(save_folder, video_filename_proc)):
            return True

        video_filename_orig = video_base_filename + '_orig.mp4'

        try:
            yt = YouTube(youtube_download_link, use_oauth=True, allow_oauth_cache=True)
        except VideoUnavailable:
            print(f"Video {video_base_filename} unavailable, skipping")
            return False

        try:  # downloading the video
            print(f"Downloading {video_base_filename} at link {youtube_download_link}"
                  f"to {os.path.join(save_folder, video_filename_orig)}")
            yt.streams.filter(
                progressive=True, file_extension='mp4', resolution="360p") \
                .first().download(output_path=save_folder, filename=video_filename_orig)
        except VideoUnavailable:
            print(f"Download Error! URL: {youtube_download_link}")
            return False

        # cropping the videos as with timestamps from the dataset
        orig_path = os.path.join(save_folder, video_filename_orig)
        proc_path = os.path.join(save_folder, video_filename_proc)

        ff.options(f"-y -v quiet -i '{orig_path}' -ss {t_beg} -t {t_end - t_beg} '{proc_path}'")
        if not os.path.exists(proc_path):
            raise RuntimeError("FFmpeg command did not produce a processed video file")
        os.remove(os.path.join(save_folder, video_filename_orig))
        return True

    def __pickle_videos(self):

        download_folder = os.path.join(self.raw_folder, self.split)
        pickles_folder = os.path.join(self.processed_folder, self.split)

        self.__makedir_exist_ok(pickles_folder)

        # Main block
        for cls in self.classes:
            cls_path = os.path.join(download_folder, cls)
            dataset = []
            print(f'Pickling {self.split}: {cls} samples')
            for vid in sorted(os.listdir(cls_path)):
                if not vid.endswith('.mp4'):
                    continue
                # Check correct file type
                retry_flag = False  # Retrial flag for when cv2.frame_count is inconsistent
                first_pass = True  # First pass flag for the current video sample
                frame_counter = 1
                while (retry_flag or first_pass):
                    first_pass = False
                    vid_path = os.path.join(cls_path, vid)
                    cap = cv2.VideoCapture(vid_path)
                    if cap.isOpened():

                        if retry_flag:
                            vidF = frame_counter  # Actual number of frames
                            retry_flag = False
                        else:
                            vidF = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # No. frames given by cv2

                        if vidF < 2*self.num_frames_dataset:
                            print(f'Insufficient frame number ({vidF}<{self.num_frames_dataset})',
                                  f'skipping {vid}')
                        elif vidF > 12*self.num_frames_dataset:
                            print(f'Too many frames ({vidF}) skipping {vid}')
                        else:  # Sample fixed number of frames from whole video
                            frame_idx = np.linspace(1, vidF, self.num_frames_dataset,
                                                    dtype=np.uint32)

                            # Start sampling frames
                            vidFrames = []
                            is_read, frame = cap.read()
                            while is_read:
                                if frame_counter in frame_idx:
                                    is_resized, frame_resized = \
                                        self.__adjust_img(frame, self.img_size)
                                    if is_resized:
                                        vidFrames.append(frame_resized)  # Successfully resized
                                    else:
                                        print(f'W - Frame {frame_counter} not resized correctly',
                                              f' with shape {frame_resized.shape}!')
                                is_read, frame = cap.read()
                                frame_counter += 1

                            frame_counter -= 1  # Correct number of frames, will be used if retry
                            if frame_counter < vidF:
                                print(f'W - Cannot read all the frames of {vid}, retrying.')
                                retry_flag = True
                            elif len(vidFrames) == self.num_frames_dataset:  # Successfully sampled
                                dataset_index = int(vid.split('_')[1])  # Video file index
                                dataset.append((vidFrames, cls, dataset_index))
                            else:
                                print(f'W - Video {vid} is not sampled correctly',
                                      f' with {len(vidFrames)} frames')
                    else:
                        print(f'W - Cannot open and skipping {vid}')
                        retry_flag = False
                    cap.release()
            if len(dataset) > 0:
                self.__write_pickle(cls, dataset, pickles_folder)

        dataset = []

    # Write pkl file with set and class name
    def __write_pickle(self, class_name, dataset, path):
        class_name = class_name.replace(' ', '_')
        num_vids = len(dataset)
        filename = f'{self.split}_{class_name}_{num_vids}samples.pkl'
        with open(os.path.join(path, filename), "wb") as output_file:
            print(f'I - Writing pickle {filename}')
            pickle.dump(dataset, output_file)

    def __adjust_img(self, image, target_img_size):
        # Center crop the frames & resize, paying attention to the short edge
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_size = image.shape
        rat0 = target_img_size[0] / img_size[0]
        rat1 = target_img_size[1] / img_size[1]
        resize_ratio = max(rat0, rat1)
        img_resized = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio,
                                 interpolation=cv2.INTER_CUBIC)
        min_x = (img_resized.shape[0] - target_img_size[0]) // 2
        max_x = min_x + target_img_size[0]
        min_y = (img_resized.shape[1] - target_img_size[1]) // 2
        max_y = min_y + target_img_size[1]
        img_resized = img_resized[min_x:max_x, min_y:max_y, :]
        if img_resized.shape == (target_img_size[0], target_img_size[1], 3):  # Check correct size
            return True, img_resized
        return False, None

    # Main dataloader function
    def __load_dataset(self):

        # Load blacklist entries
        if self.blacklist_file is not None:
            with open(os.path.join(self.root, 'kinetics400', 'blacklists', self.blacklist_file),
                      encoding='utf-8') as stream:
                blacklist_dict = yaml.load(stream, Loader=yaml.FullLoader)
                self.blacklist = [item for sublist in list(blacklist_dict.values())
                                  for item in sublist]
                print(f'Blacklist loaded with {len(self.blacklist)} entries')
        else:
            self.blacklist = []

        # Load dataset samples
        self.folder_path = os.path.join(self.processed_folder, self.split)
        dir_contents = sorted(os.listdir(self.folder_path))
        dir_pickles = [x for x in dir_contents if x.endswith('.pkl')]  # Use only pickle files
        if len(dir_pickles) != len(self.classes):
            raise ValueError(f'Number of processed class files ({len(dir_pickles)}) does'
                             f'not match with number of expected classes ({len(self.classes)})')
        print("I - ==========", self.split.upper(), "SET ==========")

        for pickle_filename in dir_pickles:
            print(f'I - Loading file: {pickle_filename} in {self.folder_path}')
            pickle_filepath = os.path.join(self.folder_path, pickle_filename)
            with open(pickle_filepath, 'rb') as f:
                dataset = pickle.load(f)
                self.add_samples(dataset)

    # Size of dataset
    def __len__(self):
        return len(self.dataset)

    # Item loader during epochs
    def __getitem__(self, index):
        # return a video of length num_frames_model, from a random starting frame

        (imgs, lab, _) = self.dataset[index]

        start_ind = np.random.randint(low=0, high=len(imgs) - self.num_frames_model + 1)
        images = imgs[start_ind:start_ind+self.num_frames_model]

        transforms_album = []
        if self.augmentation:
            transforms_album.append(A.RandomResizedCrop(height=images[0].shape[0],
                                                        width=images[0].shape[0],
                                                        scale=(0.5, 1.0),
                                                        ratio=(0.75, 1.3333333333333333),
                                                        p=1.0))
            transforms_album.append(A.HorizontalFlip(p=0.5))

            transform_album = A.Compose(transforms_album, additional_targets={
                'image0': 'image',
                'image1': 'image',
                'image2': 'image',
                'image3': 'image',
                'image4': 'image',
                'image5': 'image',
                'image6': 'image',
                'image7': 'image',
                'image8': 'image',
                'image9': 'image',
                'image10': 'image',
                'image11': 'image',
                'image12': 'image',
                'image13': 'image',
                'image14': 'image'
            })

            images_transformed = transform_album(
                image=images[0],
                image0=images[1],
                image1=images[2],
                image2=images[3],
                image3=images[4],
                image4=images[5],
                image5=images[6],
                image6=images[7],
                image7=images[8],
                image8=images[9],
                image9=images[10],
                image10=images[11],
                image11=images[12],
                image12=images[13],
                image13=images[14],
                image14=images[15]
            )

            for x in range(0, len(images_transformed)):
                if not x:
                    images[0] = images_transformed['image']
                else:
                    images[x] = images_transformed['image' + str(x - 1)]

        images = [self.__normalize_image(img) for img in images]  # Normalize images
        images_concat = []

        if self.transform is not None:
            images = [self.transform(img).type(torch.float) for img in images]
            images_concat = [torch.cat((x, y), dim=0) for x, y in zip(images, images[1:])]
            images_final = torch.stack(images_concat)
        else:  # No transform
            images_concat = [np.concatenate((x, y), axis=2) for x, y in zip(images, images[1:])]
            images_final = \
                torch.Tensor(np.array(images_concat).transpose((0, 3, 1, 2))).type(torch.float)

        return images_final, torch.tensor(lab, dtype=torch.long)

    @staticmethod
    def __normalize_image(image):
        return image / 255

    def add_samples(self, dataset, blacklist_flag=True):
        """Adds video samples to dataset"""
        for data in dataset:

            (imgs, lab, vidx) = data
            lab = self.classes.index(lab)
            if vidx in self.blacklist and blacklist_flag:
                continue  # Blacklist sample

            if len(imgs) > self.num_frames_dataset:  # Check correct frame count
                print("I - Number of frames greater than dataset description.")
                continue

            imgs = imgs[1:-1]  # Toss first and last frames
            len_imgs = len(imgs)
            if len_imgs < self.num_frames_model:  # Check sufficient frame count
                print("I - Tossed video with insufficient number of frames.")
                continue

            if self.split == 'train':
                self.dataset.append((imgs, lab, vidx))
            else:  # get 3 fixed segments from each video for validation and test
                self.dataset.append((imgs[0:self.num_frames_model], lab, vidx))  # beginning
                self.dataset.append((imgs[len_imgs//2-self.num_frames_model//2:
                                    len_imgs//2+self.num_frames_model//2], lab, vidx))  # middle
                self.dataset.append((imgs[-self.num_frames_model:], lab, vidx))  # end


def kinetics_get_datasets(
        data, load_train=True, load_test=True, num_classes=4,
        img_size=(240, 240), fold_ratio=4, num_frames_model=16, num_frames_dataset=50,
        max_train_examples_per_class=2000, max_test_examples_per_class=150):
    """
    Load the folded 16-frame version of selected classes from the Kinetics 400 dataset

    The dataset is loaded from the archive file from DeepMind.

    The dataset originally includes 400 action classes. This dataset is formed with 5 classes which
    includes 4 of the action classes. The rest of the dataset is used to form the "other" class.

    The training data is augmented by:
    - random spatial cropping of videos (with sensible parameters)
    - flipping them horizontally with 50% chance
    - randomly selecting segments of 16 frames from the videos that are originally 50 frames long

    The test data uses 3 consecutive fixed segments from each video, with 16 frames each.

    The current implementation of using 2000 training and 150 test examples per class
    at 240x240 resolution and 5 frames per second requires around 50 GB of RAM.
    """
    (data_dir, args) = data

    transform = transforms.Compose([
        transforms.ToTensor(),
        ai8x.normalize(args=args),
        ai8x.fold(fold_ratio=fold_ratio)
    ])

    if num_classes == 4:
        classes = next((e for _, e in enumerate(datasets)
                        if len(e['output']) - 1 == num_classes))['output']
    else:
        raise ValueError(f'Unsupported num_classes {num_classes}')

    if load_train:
        train_dataset = Kinetics(root=data_dir, split='train', img_size=img_size, classes=classes,
                                 num_frames_model=num_frames_model,
                                 num_frames_dataset=num_frames_dataset,
                                 max_examples_per_class=max_train_examples_per_class,
                                 transform=transform, augmentation=True, download=True)
    else:
        train_dataset = None

    if load_test:
        test_dataset = Kinetics(root=data_dir, split='test', img_size=img_size, classes=classes,
                                num_frames_model=num_frames_model,
                                num_frames_dataset=num_frames_dataset,
                                max_examples_per_class=max_test_examples_per_class,
                                transform=transform, augmentation=False, download=True)

        if args.truncate_testset:
            test_dataset.dataset = test_dataset.dataset[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'Kinetics400',
        'input': (96, 60, 60),
        'output': ('pull ups', 'push up', 'situp', 'squat', 'other'),
        'weight': (0.17, 0.325, 0.215, 0.17, 0.12),
        'loader': kinetics_get_datasets,
        'fold_ratio': 4,
    },  # make sure the class names in "output" match those the kinetics dataset, except 'other'
]
