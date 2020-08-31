###################################################################################################
#
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""evaluate an onnex model with the model test dataset
"""
from datetime import datetime
from pydoc import locate
import sys
import random as rn
import os
import fnmatch
import argparse
import tensorflow as tf
import numpy as np
import onnxruntime as rt

# following piece it to init seed to make reproducable results
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(10)
rn.seed(100)
tf.random.set_seed(7)

VERBOSE = 1

# command parser
parser = argparse.ArgumentParser(description='set input arguments')

parser.add_argument(
    '--onnx-file',
    required=True,
    dest='onnx_file',
    type=str,
    help='qunatized or unquantized onnx file')
parser.add_argument(
    '--dataset',
    required=True,
    dest='dataset',
    type=str,
    help='dataset for the model')
parser.add_argument(
    '--inputs-as-nchw',
    action='store_true',
    required=False,
    dest='nchw',
    default=False,
    help='onnx model input is nchw (default:false)')
args = parser.parse_args()


# create a class for logging screen to file
# make sure to change verbose=2 in fit and evaluate
class Logger():
    """
    stdout logger
    """

    def __init__(self, filename):
        """
        init
        """
        self.terminal = sys.stdout
        self.filename = filename
        self.log = None

    def write(self, message):
        """
        write to file
        """
        with open(self.filename, "a") as self.log:
            self.log.write(message)
        self.terminal.write(message)

    def flush(self):
        """
        this flush method is needed for python 3 compatibility.
        this handles the flush command by doing nothing.
        you might want to specify some extra behavior here.
        """
        pass  # pylint: disable=unnecessary-pass


if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)

    onnx_file = args.onnx_file
    model_dataset = args.dataset
    nchw = args.nchw

    # Log stdout to file
    foldername = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join('logs', foldername + '-evaluate-' + model_dataset)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    sys.stdout = Logger(os.path.join(logdir,  # type: ignore[assignment] # noqa: F821
                                     foldername + '.log'))

    print('script:', sys.argv[0])
    print('log dir:', logdir)

    # Dynamically load datasets
    for _, _, files in sorted(os.walk('datasets')):  # type: ignore[assignment] # noqa: F821
        for name in sorted(files):
            if model_dataset == name[:-3] and fnmatch.fnmatch(name, '*.py'):
                try:
                    ds = locate('datasets.' + name[:-3])

                    # load data set
                    (train_images, train_labels), (test_images, test_labels), (
                        valid_images, valid_labels) = \
                        ds.get_datasets('./data')  # type: ignore[attr-defined] # noqa: F821

                    class_names = ds.get_classnames()  # type: ignore[attr-defined] # noqa: F821
                    print('Class Names:', class_names)
                except AttributeError:
                    raise RuntimeError("cannot load" + model_dataset)
                break

    print('Dataset:', ds)

    print("test_images shape:", test_images.shape)
    print("test_labels shape:", test_labels.shape)

    # Normalize data to [-0.5, 0.5] range
    print('Normalize image to [-0.5,0.5] range')
    train_images = train_images/256.0
    valid_images = valid_images/256.0
    test_images = test_images/256.0

    if nchw:
        ndim = train_images.ndim
        print(ndim)
        if ndim in (3, 4):
            train_images = train_images.swapaxes(1, ndim-1)
            valid_images = valid_images.swapaxes(1, ndim-1)
            test_images = test_images.swapaxes(1, ndim-1)

            train_images = train_images.swapaxes(ndim-1, ndim-2)
            valid_images = valid_images.swapaxes(ndim-1, ndim-2)
            test_images = test_images.swapaxes(ndim-1, ndim-2)
        else:
            print("Error: ndim should be 3 or 4!")
            sys.exit(0)

    # Inference session
    sess = rt.InferenceSession(onnx_file)

    correct_count = 0
    for index in range(test_images.shape[0]):
        input_image = np.expand_dims(
            np.array(test_images[index], dtype=np.float32), 0)

        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        # Run model on image data
        result = sess.run([output_name], {input_name: input_image})
        # print ( result )
        predict = np.argmax(result)
        # print('[%d]Actual:%d Pred: %d\nConf: %f'
        # %(index,test_labels[index], predict , result[0][0][predict]))

        if predict == test_labels[index]:
            correct_count += 1

        if index % 100 == 0:
            print(f'{index}:, test_accuracy: {correct_count / (index + 1)}')

    print(
        f'\nTotal number: {index+1}, test_accuracy: {correct_count/(index+1)}')
