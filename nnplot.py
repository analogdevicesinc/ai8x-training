###################################################################################################
#
# Copyright (C) 2018-2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
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
                      newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    buf.close()
    return data
