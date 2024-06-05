#
# MIT License
#
# Copyright (c) 2020 Alex
# Portions Copyright (C) 2024 Analog Devices, Inc.
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
"""
Python implementation of MATLAB's image resize function imresize()
Source: https://github.com/fatheral/matlab_imresize
Modification made to refactor the code and add comments.
"""

from __future__ import print_function

import sys
from math import ceil

import numpy as np


def deriveSizeFromScale(img_shape, scale):
    """Function to calculate output shape from input shape and expected scale"""
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape


def deriveScaleFromSize(img_shape_in, img_shape_out):
    """Function to calculate scale from input shape and output shape"""
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale


def triangle(x):
    """Triangle function"""
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x >= -1), x < 0)
    greaterthanzero = np.logical_and((x <= 1), x >= 0)
    f = np.multiply((x+1), lessthanzero) + np.multiply((1-x), greaterthanzero)
    return f


def cubic(x):
    """Cubic function"""
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + \
        np.multiply(-0.5*absx3 + 2.5*absx2 - 4*absx + 2, (1 < absx) & (absx <= 2))
    return f


def contributions(in_length, out_length, scale, kernel, k_width):
    """Contributions function"""
    if scale < 1:
        def h(x):
            return scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width

    x = np.arange(1, out_length+1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1  # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1)  # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1)))
    aux = aux.astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]

    return weights, indices


def imresizemex(img_in, weights, indices, dim):
    """imresizemex function"""
    in_shape = img_in.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    img_out = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = img_in[ind, i_img].astype(np.float64)
                img_out[i_w, i_img] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T),
                                             axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = img_in[i_img, ind].astype(np.float64)
                img_out[i_img, i_w] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T),
                                             axis=0)

    if img_in.dtype == np.uint8:
        img_out = np.clip(img_out, 0, 255)
        return np.around(img_out).astype(np.uint8)

    return img_out


def imresizevec(img_in, weights, indices, dim):
    """imresizevec function"""
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        img_out = np.sum(weights*((img_in[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        img_out = np.sum(weights*((img_in[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)

    if img_in.dtype == np.uint8:
        img_out = np.clip(img_out, 0, 255)
        return np.around(img_out).astype(np.uint8)

    return img_out


def resizeAlongDim(A, dim, weights, indices, mode="vec"):
    """Resize image wrt input mode"""
    if mode == "org":
        out = imresizemex(A, weights, indices, dim)
    else:
        out = imresizevec(A, weights, indices, dim)
    return out


def imresize(img, scalar_scale=None, method='bicubic', output_shape=None, mode="vec"):
    """Function to resize image"""
    if method == 'bicubic':
        kernel = cubic
    elif method == 'bilinear':
        kernel = triangle
    else:
        print('Error: Unidentified method supplied')

    kernel_width = 4.0
    # Fill scale and output_size
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(img.shape, scale)
    elif output_shape is not None:
        scale = deriveScaleFromSize(img.shape, output_shape)
        output_size = list(output_shape)
    else:
        print('Error: scalar_scale OR output_shape should be defined!')
        sys.exit()

    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = contributions(img.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(img)
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B


def convertDouble2Byte(img):
    """Function to convert double data to byte"""
    img_out = np.clip(img, 0.0, 1.0)
    img_out = 255*img_out
    return np.around(img_out).astype(np.uint8)
