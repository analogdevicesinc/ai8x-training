#
# MIT License
#
# Copyright (c) 2019 Sagar Vinodababu
# Copyright (c) 2017 Max deGroot, Ellis Brown
# Portions Copyright (C) 2023 Maxim Integrated Products, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Helper methods originate from:
# https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
# Some augmentation functions have been adapted from:
# https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
#

""" Some utility functions for Augmentation Tasks """
import random

import torch
import torchvision.transforms.functional as FT

from utils import object_detection_utils


def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.

    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ == 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18
                # #we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5
                # for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image


def expand(image, boxes, filler):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.

    Helps to learn to detect smaller objects.

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param filler: RBG values of the filler material, a list like [R, G, B]
    :return: expanded image, updated bounding box coordinates
    """
    # Calculate dimensions of proposed expanded (zoomed-out) image
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)

    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler
    filler = torch.tensor(filler, dtype=torch.float)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * \
        filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)

    # Note - do not use expand() like
    # new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # because all expanded values will share the same memory,
    # so changing one pixel will change all

    # Place the original image at random coordinates in this new image
    # (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Adjust bounding boxes' coordinates accordingly
    # (n_objects, 4), n_objects is the no. of objects in this image
    new_boxes = \
        boxes + torch.tensor([left, top, left, top], dtype=torch.float).unsqueeze(0)

    return new_image, new_boxes


def flip(image, boxes):
    """
    Flip image horizontally.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    """
    # Flip image
    new_image = FT.hflip(image)

    # Flip boxes
    new_boxes = boxes
    # new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    # new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes[:, 0] = image.shape[2] - boxes[:, 0] - 1
    new_boxes[:, 2] = image.shape[2] - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def random_crop(image, boxes, labels, difficulties):
    """
    Performs a random crop in the manner stated in the paper.
    Helps to learn to detect larger and partial objects.

    Note that some objects may be cut out entirely.

    Adapted from
    https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates,
    a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects,
    a tensor of dimensions (n_objects)
    :return: cropped image, updated bounding box coordinates, updated labels,
    updated difficulties
    """
    original_h = image.size(1)
    original_w = image.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        # 'None' refers to no cropping
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])

        # If not cropping
        if min_overlap is None:
            return image, boxes, labels, difficulties

        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is
        # chosen in paper authors' original Caffe repo
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in
            # the authors' repo
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.tensor([left, top, right, bottom], dtype=torch.float)  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            # (1, n_objects), n_objects is the no. of objects in this image
            overlap = \
                object_detection_utils.find_jaccard_overlap(crop.unsqueeze(0),
                                                            boxes)
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than
            # the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * \
                              (bb_centers[:, 1] > top) * (bb_centers[:, 1] < bottom)

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            # crop[:2] is [left, top]
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])
            new_boxes[:, :2] -= crop[:2]
            # crop[2:] is [right, bottom]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties


def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    """
    Resize image. For the SSD300, resize to (300, 300).
    Since percent/fractional coordinates are calculated for the bounding boxes
    (w.r.t image dimensions) in this process,
    you may choose to retain them.
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions
    (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates,
    in which case they remain the same)
    """
    # Resize image
    new_image = FT.resize(image, dims, antialias=True)

    # Resize bounding boxes
    # old_dims =
    # torch.tensor([image.width, image.height, image.width, image.height],
    #              dtype=torch.float).unsqueeze(0)
    old_dims = torch.tensor(
        [image.shape[2], image.shape[1], image.shape[2], image.shape[1]],
        dtype=torch.float).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.tensor([dims[1], dims[0], dims[1], dims[0]],
                                dtype=torch.float).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes
