#!/usr/bin/env python3
###################################################################################################
#
# Copyright (C) 2020-2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Script to generate dataset for FaceID training and validation from YouTubeFaces dataset.
"""

import argparse
import json
import os

import numpy as np
import torch

import scipy.ndimage
from facenet_pytorch import MTCNN, InceptionResnetV1  # pylint: disable=no-name-in-module
from matplotlib.image import imread


def generate_image(img, box, count):  # pylint: disable=too-many-locals
    """
    Generates images in size 120x160x3 that includes the detected face in the image.

    img, box are the original image and box.
    count is how many pics you wanna generate

    box format: x1, y1, x3, y3
    """
    box[0] = np.max((box[0], 0))
    box[1] = np.max((box[1], 0))
    box[2] = np.min((box[2], img.shape[1]))
    box[3] = np.min((box[3], img.shape[0]))

    factor = 1
    height = img.shape[0]
    width = img.shape[1]
    new_img = img
    new_box = box
    while True:
        if height < 160 or width < 120:
            factor += 1
            new_img = scipy.ndimage.zoom(img, [factor, factor, 1], order=1)
            new_box = box * factor
            height = new_img.shape[0]
            width = new_img.shape[1]
        else:
            break
    new_box = np.round(new_box).astype(np.int)
    new_box_height = new_box[3] - new_box[1]
    new_box_width = new_box[2] - new_box[0]

    scale_list = np.arange(0.9, 0, -0.1)
    ind = 0
    while (new_box_height > 160 or new_box_width > 120):
        new_img = scipy.ndimage.zoom(img, [scale_list[ind], scale_list[ind], 1], order=1)
        new_box = box * scale_list[ind]
        new_box = np.round(new_box).astype(np.int)
        new_box_height = new_box[3] - new_box[1]
        new_box_width = new_box[2] - new_box[0]
        ind += 1

    min_x = np.max((0, new_box[0] - (120 - new_box_width)))
    min_y = np.max((0, new_box[1] - (160 - new_box_height)))
    max_x = np.min((new_box[0], width-120))
    max_y = np.min((new_box[1], height-160))

    start_x = np.random.choice(np.arange(min_x, max_x+1), count, replace=True)
    start_y = np.random.choice(np.arange(min_y, max_y+1), count, replace=True)
    img_arr = []
    box_arr = []
    for i in range(count):
        img_arr.append(new_img[start_y[i]:start_y[i]+160, start_x[i]:start_x[i]+120])
        temp_box = new_box.copy()
        temp_box[0] -= start_x[i]
        temp_box[2] -= start_x[i]
        temp_box[1] -= start_y[i]
        temp_box[3] -= start_y[i]
        box_arr.append(temp_box)
    new_img = img_arr
    new_box = box_arr
    return new_img, new_box, img, box


def main(source_path, dest_path):
    """
    Main function to iterate over the images in the raw data and generate data samples
    to train/test FaceID model.
    """

    # img_dir = os.path.join(raw_data_path, 'aligned_images_DB')
    frame_dir = os.path.join(source_path, 'frame_images_DB')

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    # set parameters
    num_imgs_per_face = 1
    target_im_shape = (160, 120)

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    # create models
    mtcnn = MTCNN(
        image_size=80, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # run models on the images
    num_persons = 0
    num_faces = 0

    embedding_dict = {}
    subj_name_list = os.listdir(frame_dir)

    for f_n, face_file in enumerate(subj_name_list):
        if (f_n % 100) == 0:
            print(f'Subject {f_n} of {len(subj_name_list)}')
        f_path = os.path.join(frame_dir, face_file)
        if os.path.isfile(f_path):
            if face_file.endswith('txt'):
                with open(f_path, mode='r', encoding='utf-8') as file:
                    lines = file.readlines()
                    num_persons += 1
                    for line in lines:
                        num_faces += 1
                        img_name = line.split(',')[0]
                        subj_name, video_no, file_name = img_name.split('\\')
                        img_path = os.path.join(frame_dir, subj_name, video_no, file_name)
                        img = imread(img_path)

                        x_aligned, _, _ = mtcnn(img, return_prob=True)
                        if x_aligned is not None:
                            aligned = x_aligned[None, :, :, :].to(device)
                            embedding = resnet(aligned).detach().cpu().numpy()[0]

                            if subj_name not in embedding_dict:
                                embedding_dict[subj_name] = {}
                                subj_path = os.path.join(dest_path, subj_name)
                                if not os.path.exists(subj_path):
                                    os.mkdir(subj_path)
                            if video_no not in embedding_dict[subj_name]:
                                embedding_dict[subj_name][video_no] = {}
                                video_path = os.path.join(dest_path, subj_name, video_no)
                                if not os.path.exists(video_path):
                                    os.mkdir(video_path)

                            embedding_dict[subj_name][video_no][file_name] = embedding.tolist()
                            x_aligned_int = x_aligned.cpu().numpy()
                            x_aligned_int -= np.min(x_aligned_int)
                            x_aligned_int /= np.max(x_aligned_int)
                            x_aligned_int = (255.0 * x_aligned_int).astype(np.uint8)
                            np.save(os.path.join(dest_path, subj_name, video_no, file_name),
                                    x_aligned_int)

                            rect = line.split(',')[2:6]
                            for i in range(4):
                                rect[i] = int(rect[i])

                            box = np.array([int(rect[0]) - int(rect[2])//2,
                                            int(rect[1]) - int(rect[3])//2,
                                            int(rect[0]) + int(rect[2])//2,
                                            int(rect[1]) + int(rect[3])//2])

                            img_arr, _, img, box = generate_image(img, box, num_imgs_per_face)
                            for img_idx in range(num_imgs_per_face):
                                new_file_name = '_'.join([file_name, str(target_im_shape[0]),
                                                          str(target_im_shape[1]), str(img_idx)])
                                cropped_im_path = os.path.join(dest_path, subj_name, video_no,
                                                               new_file_name)
                                np.save(cropped_im_path, img_arr[img_idx])

    print(f'Number of People: {num_persons}')
    print(f'Number of Faces: {num_faces}')

    # save embeddings to json file
    with open(os.path.join(dest_path, 'embeddings.json'), mode='w', encoding='utf-8') as out_file:
        json.dump(embedding_dict, out_file)


def parse_args():
    """Parses command line arguments"""
    data_folder = os.path.abspath(__file__)
    for _ in range(3):
        data_folder = os.path.dirname(data_folder)
    data_folder = os.path.join(data_folder, 'data')

    parser = argparse.ArgumentParser(description='Generate YouTubeFaces dataset to train/test \
                                                  FaceID model.')
    parser.add_argument('-r', '--raw', dest='raw_data_path', type=str,
                        default=os.path.join(data_folder, 'YouTubeFaces', 'raw'),
                        help='Path to raw YouTubeFaces dataset folder.')
    parser.add_argument('-d', '--dest', dest='dest_data_path', type=str,
                        default=os.path.join(data_folder, 'YouTubeFaces'),
                        help='Folder path to store processed data')
    parser.add_argument('--type', dest='data_type', type=str, required=True,
                        help='Data type to generate (train/test)')
    args = parser.parse_args()

    source_path = args.raw_data_path
    dest_path = os.path.join(args.dest_data_path, args.data_type, 'temp')
    return source_path, dest_path


if __name__ == "__main__":
    raw_data_path, dest_data_path = parse_args()
    main(raw_data_path, dest_data_path)
