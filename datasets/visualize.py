###################################################################################################
#
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Default data visualization
"""


def visualize_data(data, args):
    """
    Default data visualization. Converts 2D data in color (RGB) or monochrome format to image.
    Input: `data`. The first dimension is the batch dimension, the second the channel, followed
    by the rest of the data.
    For example, a batch of RGB images would be NCHW with C=3, and C=1 for monochrome images.
    `args` are additional arguments.
    """
    if len(data.shape) == 4 and data.shape[1] in [1, 3] and data.shape[2] == data.shape[3]:
        # Only add data for 2D, RGB or monochrome data when width == height
        if args.act_mode_8bit:
            data /= 255.0  # Scale for display

        return data

    return None  # Don't display data since we don't know how
