###################################################################################################
#
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""train a keras sequential model
"""
from datetime import datetime
from pydoc import locate
import sys
import random as rn
from random import randint
import os
import fnmatch
import argparse
import tensorflow as tf
import numpy as np

# following piece it to init seed to make reproducable results
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(10)
rn.seed(100)
tf.random.set_seed(7)

VERBOSE = 2

# command parser
parser = argparse.ArgumentParser(description='set input arguments')

parser.add_argument(
    '--epochs',
    action='store',
    dest='epochs',
    type=int,
    default=100,
    help='number of total epochs to run (default: 100)')
parser.add_argument(
    '--batch_size',
    action='store',
    dest='batch_size',
    type=int,
    default=32,
    help='Size of the training batch')
parser.add_argument(
    '--model', required=True, dest='model', type=str, help='CNN model')
parser.add_argument(
    '--dataset',
    required=True,
    dest='dataset',
    type=str,
    help='dataset for the model')
parser.add_argument(
    '--optimizer',
    default='Adam',
    dest='optimizer',
    type=str,
    help='optimizer for training (default: Adam)')
parser.add_argument(
    '--lr',
    default=0.0001,
    dest='lr',
    type=float,
    help='initial learning rate')
parser.add_argument(
    '--save-sample',
    action='store',
    dest='generate_sample',
    type=int,
    help='save the sample at given index as NumPy sample data')
parser.add_argument(
    '--metrics',
    dest='metrics',
    default='accuracy',
    type=str,
    help='metrics used in compiling model(default: accuracy)')
args = parser.parse_args()

# parser.print_help()
# print('input args: ', args)


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

    epochs = args.epochs
    batch_size = args.batch_size
    cnn_model = args.model
    optimizer_type = args.optimizer
    learningrate = args.lr
    model_dataset = args.dataset
    model_optimizer = args.optimizer
    sample_index = args.generate_sample
    metrics = args.metrics

    # Log stdout to file
    foldername = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join('logs', foldername + '-' + model_dataset)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    sys.stdout = Logger(os.path.join(logdir,  # type: ignore[assignment] # noqa: F821
                                     foldername + '.log'))
    print('script:', sys.argv[0])
    print('log dir:', logdir)
    print('epochs:', epochs)
    print('batch_size:', batch_size)

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

                    class_names = ds.get_classnames()   # type: ignore[attr-defined] # noqa: F821
                    print('Class Names:', class_names)
                except AttributeError:
                    raise RuntimeError("cannot load" + model_dataset)
                break

    # Dynamically load model
    for _, _, files in sorted(os.walk('models')):  # type: ignore[assignment] # noqa: F821
        for name in sorted(files):
            if cnn_model == name[:-3] and fnmatch.fnmatch(name, '*.py'):
                try:
                    md = locate('models.' + name[:-3])

                    # printing model
                    print('\n' + '-' * 20 + 'Model:' + cnn_model + '-' * 20)
                    with open(os.path.join('models', name)) as fin:
                        for line in fin:
                            if not line.startswith("model"):
                                continue
                            print(line.rstrip())
                            for l in fin:
                                print(l.rstrip())
                    print('-' * 50)

                except AttributeError:
                    raise RuntimeError("cannot load" + cnn_model)
                break

    print('Model:', cnn_model)
    print('optimizer:', optimizer_type)
    print('Initial lr:', learningrate)
    print('Metrics:', metrics)
    print('Dataset:', ds)

    print("train_images shape:", train_images.shape)
    print("valid_images shape:", valid_images.shape)
    print("test_images shape:", test_images.shape)

    print("train_labels shape:", train_labels.shape)
    print("valid_labels shape:", valid_labels.shape)
    print("test_labels shape:", test_labels.shape)

    # print("train_images min:",train_images.min())
    # print("train_image max:",train_images.max())
    # print("train_labels min:", train_labels.min())
    # print("train_labels max:", train_labels.max())

    # Create a custom learning rate
    lr_init = learningrate  # needed for clamping
    print("Init LR:", lr_init)

    # callbacks
    callbacks = [
        md.lr_schedule,  # type: ignore[attr-defined] # noqa: F821
        # TensorBoard(log_dir=logdir),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(logdir, 'checkpoint-best.hdf5'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max')
    ]

    # Build the tf.keras model using the Keras model subclassing API:
    model = md.model  # type: ignore[attr-defined] # noqa: F821

    # select optimizer
    if model_optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(lr=lr_init, beta_1=0.9,
                                             beta_2=0.999, epsilon=1e-07)
    elif model_optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD(lr=lr_init, momentum=0.0)
    else:
        raise RuntimeError('optimizer not implemented!')

    # compile model with optimizer
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[metrics])

    # Train model, specify number of epochs and batch size
    history = model.fit(
        train_images,
        train_labels,
        validation_data=(valid_images, valid_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=VERBOSE)

    # Evaluate model with test set
    test_loss, test_acc = model.evaluate(
        test_images, test_labels, verbose=VERBOSE)

    print("Test Accuracy:", test_acc)

    # print confusion matrix
    predict = np.argmax(model.predict(test_images), axis=1)
    print("Confusion Matrix:")
    print(tf.math.confusion_matrix(test_labels, predict))

    print("Check prediction for some random indexes:")
    for i in range(10):
        index = randint(0, test_labels.size)
        print("\tindex: %d: predict: %d   actual: %d" % (index, predict[index],
                                                         test_labels[index]))

    # save one sample from test set
    if sample_index:
        index = sample_index
        # Adjust the shape similar to the model shape
        print("Save sample data:")
        sample_image = np.expand_dims(
            np.array(test_images[index], dtype=np.float32), 0)
        prediction = model.predict(sample_image)
        print('\tSample index:', index)
        print('\tPrediction:', prediction)
        print(
            f'\tPredicted: {np.argmax(prediction)}({class_names[np.argmax(prediction)]})'
        )
        print(
            f'\tExpected: {test_labels[index]}({class_names[test_labels[index]]})'
        )
        np.savez(
            logdir + '/sampledata_class_' +
            f"{test_labels[index]}_all_predictions",
            sample_image=sample_image,
            prediction=prediction)
        # save as pty
        np.save(logdir +
                '/sampledata_class-' + f"{test_labels[index]}_pred-{np.argmax(prediction)}_HWC",
                np.array(sample_image, dtype=np.int32))

    # verify
    '''
    a = np.load(logdir + '/sampledata_class_'+ f"{test_labels[index]}_all_predictions" + '.npz')
    imageload = a['sample_image']
    predictionload = a['prediction']

    if not(np.array_equal(sample_image, imageload)):
          print("Error")

    if not(np.array_equal(prediction, predictionload)):
          print("Error")
    '''  # pylint: disable=pointless-string-statement

    # print model
    model.summary()

    # weights range
    print("Weight range:")
    for i, layer in enumerate(model.layers):
        weight = (layer.get_weights()[0:1])  # weights
        bias = (layer.get_weights()[1:2])  # bias
        print("\t[%d]-%s:\t Wmin= %.4f, Wmax= %.4f, Bmin= %.4f, Bmax= %.4f" %
              (i, layer.get_config()['name'], tf.math.reduce_min(weight),
               tf.math.reduce_max(weight), tf.math.reduce_min(bias),
               tf.math.reduce_min(bias)))

    # save a model
    saved_model_dir = os.path.join(logdir, model_dataset + '_SavedModel')
    tf.saved_model.save(model, saved_model_dir)

    # save a copy to onnx folder
    expdir = os.path.join('export', model_dataset)
    if not os.path.isdir(expdir):
        os.makedirs(expdir)
    tf.saved_model.save(model, expdir)

    # save a copy of sample data
    np.save(expdir +
            '/sampledata_class-' + f"{test_labels[index]}_pred-{np.argmax(prediction)}_HWC",
            np.array(sample_image, dtype=np.int32))

    # print graphical model
    tf.keras.utils.plot_model(
        model, to_file=os.path.join(logdir, 'kws20.png'), show_shapes=True)

    # return log folder
    sys.exit(saved_model_dir)
