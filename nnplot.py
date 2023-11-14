#
# MIT License
#
# Copyright (c) 2018 Dabi Ahn
# Portions Copyright (C) 2018-2023 Maxim Integrated Products, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
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
Graphical output routines
"""
import io
import itertools
import re
from textwrap import wrap

import numpy as np

import matplotlib.figure as matfig


def confusion_matrix(cm, labels, normalize=False):
    """
    Create confusion matrix image plot

    Parameters:
        cm                              : Confusion matrix
        labels                          : Axis labels (strings)

    Returns:
        data                            : Confusion matrix image buffer
    """
    if normalize:
        cm = cm.astype('float')*10.0 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)

    figsize = min(20, max(5, len(labels) // 2))

    fig = matfig.Figure(figsize=(figsize, figsize), dpi=96, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(cm, cmap='jet')

    strlabels = map(str, labels)
    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in strlabels]
    classes = ['\n'.join(wrap(cl, 40)) for cl in classes]

    tick_marks = np.arange(len(classes))

    FONTSIZE = 12
    ax.set_xlabel('Predicted', fontsize=FONTSIZE)
    ax.set_xticks(tick_marks)
    rotation = 90 if len(max(classes, key=len)) > 2 else 0
    ax.set_xticklabels(classes, fontsize=FONTSIZE, rotation=rotation, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('Actual', fontsize=FONTSIZE)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=FONTSIZE, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.',
                horizontalalignment='center',
                fontsize=FONTSIZE, verticalalignment='center', color='white')
    fig.set_tight_layout(True)

    buf = io.BytesIO()
    fig.savefig(buf, format='raw', dpi='figure')
    buf.seek(0)
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    data = np.reshape(np.frombuffer(buf.getvalue(), dtype=np.uint8),
                      newshape=(int(fig.bbox.bounds[3]),  # type: ignore
                                int(fig.bbox.bounds[2]), -1))  # type: ignore
    buf.close()
    return data
