###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""test Conv2DReLU->AvgPool-> Dense
"""
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.join('..', '..'))
import ai8xTF  # pylint: disable=import-error,wrong-import-order,wrong-import-position  # noqa:E402

ai8xTF.set_device(85, False, 10)


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


def clamp(x, minimum=-128, maximum=127):
    """
    clamp with max/min
    """
    return np.array(tf.clip_by_value(x, minimum, maximum))


# following piece it to init seed to make repeated results
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(10)
tf.random.set_seed(7)

logdir = 'saved_model'

if not os.path.isdir(logdir):
    os.makedirs(logdir)

# Log stdout to file
sys.stdout = Logger(os.path.join(logdir,  # type:ignore[assignment]
                                 'result.log'))  # noqa:F821

# Init input samples
test_input = np.random.normal(0, 0.5, size=(8, 8))
test_input = clamp(np.floor(test_input*128+0.5))/128.0
test_input = np.reshape(test_input, (1, 8, 8))
print('Test Input shape', test_input.shape)
print('Test Input', test_input)

# Init layer kernel

k1 = np.linspace(-0.9, 0.9, num=18, dtype=np.float32)
k1 = clamp(np.floor(k1*128+0.5))/128.0

k5 = np.linspace(-0.5, 0.5, num=490, dtype=np.float32)
k5 = clamp(np.floor(k5*128+0.5))/128.0

init_bias = np.array([-0.01, 0.01])
init_bias = clamp(np.floor(init_bias*128+0.5))/128.0
bias_initializer = tf.keras.initializers.constant(init_bias)

# Create functional model
input_layer = tf.keras.Input(shape=(8, 8))
reshape = tf.keras.layers.Reshape(target_shape=(8, 8, 1))(input_layer)

conv1 = ai8xTF.FusedConv2DReLU(
    filters=2,
    kernel_size=3,
    strides=1,
    padding_size=1,
    use_bias=False,
    kernel_initializer=tf.keras.initializers.constant(k1)
    )(reshape)
avgpool1 = ai8xTF.AvgPool2D(
    pool_size=(2, 2),
    pool_strides=1,
    )(conv1)
flat = tf.keras.layers.Flatten()(avgpool1)

output_layer = ai8xTF.FusedDense(5,
                                 wide=True, use_bias=False,
                                 kernel_initializer=tf.keras.initializers.constant(k5)
                                 )(flat)

model = tf.keras.Model(inputs=[input_layer], outputs=[conv1, avgpool1, flat, output_layer])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

for layer in model.layers:
    weight = np.array((layer.get_weights()[0:1]))  # weights
    # Convert to 8bit, round and clamp
    print('Weight(8-bit)=\n', clamp(np.floor(weight*128+0.5)))
    print(weight.shape)
    bias = np.array((layer.get_weights()[1:2]))  # bias
    # Convert to 8bit, round and clamp
    print('Bias(8-bit)=\n', clamp(np.floor(bias*128+0.5)))
    tf.print(f"Layer: {layer.get_config ()['name']} \
              Wmin: {tf.math.reduce_min(weight)}, \
              Wmax: {tf.math.reduce_max(weight)}, \
              Bias min: {tf.math.reduce_min(bias)}, \
              Bias max: {tf.math.reduce_min(bias)}")

conv1_out, avgpool1, flat_out, output = model.predict(test_input)

# Model output
print('Conv output=\n', conv1_out)
print('AvgPool output=\n', avgpool1)
print('Flat output=\n', flat_out)
print('Output=\n', output)

# Save model
tf.saved_model.save(model, 'saved_model')

# Convert to 8bit, round and clamp
saved_input = clamp(np.floor(test_input*128+0.5))
print('Input(8-bit):\n', saved_input)
print(saved_input.shape)
# Save input
np.save(os.path.join(logdir, 'input_sample_1x8x8.npy'), np.array(saved_input, dtype=np.int32))

# Convert to 8bit, round and clamp
print('Conv1 output(8-bit):\n', clamp(np.floor(conv1_out*128+0.5)))
print(conv1_out.shape)
print('AvgPool output(8-bit):\n', clamp(np.floor(avgpool1*128+0.5)))
print(avgpool1.shape)
print('Flat output(8-bit):\n', clamp(np.floor(flat_out*128+0.5)))
print(flat_out.shape)
print('Output(8-bit):\n', clamp(np.floor(output*128+0.5)))
print(output.shape)

sys.exit(0)