#!/usr/bin/python
# -*- coding: utf-8 -*-
###################################################################################################
#
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################

"""converts native TF sample data to izer expected format
"""

import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np

# command parser

parser = \
    argparse.ArgumentParser(
        description='convert sample data from Tensorflow HWC to izer CHW or CWH format'
    )
parser.add_argument('--input', required=True, dest='input', type=str,
                    help='input: TF native nhwc sample data file')
parser.add_argument('--output', required=True, dest='output', type=str,
                    help='output: PT native chw or  cwh sample data file'
                    )
parser.add_argument('--swap-output', required=False, action='store_true',
                    dest='swap', help='swap w and h in output file')
args = parser.parse_args()

if __name__ == '__main__':
    image1 = np.load(args.input)
    print('input file shape:', image1.shape)

    if image1.shape[0] != 1:
        print('size of first axes should be 1')
        sys.exit(0)

    if image1.ndim == 4:
        image1 = image1.reshape(image1.shape[1], image1.shape[2],
                                image1.shape[3])
        print('removed n:', image1.shape)
        image1 = image1.swapaxes(0, 2)
        print('converted to cwh:', image1.shape)
        if not args.swap:
            image1 = image1.swapaxes(1, 2)
            print('converted to chw:', image1.shape)

        # swap axes for display only
        tmp_img = image1.swapaxes(0, 2).swapaxes(0, 1)
        plt.imshow(tmp_img + 128, cmap='gray')
    elif image1.ndim == 3:
        image1 = image1.swapaxes(0, 2)
        print('converted to cwh:', image1.shape)
        if not args.swap:
            image1 = image1.swapaxes(0, 1)
            print('converted to chw:', image1.shape)
        plt.imshow(image1.squeeze(axis=2) + 128, cmap='gray')
    else:
        print('ERROR: Unexpected inputformat!')
        sys.exit(0)

    print('output file shape:', image1.shape)
    np.save(args.output, image1)
    plt.show()

    print(image1)
