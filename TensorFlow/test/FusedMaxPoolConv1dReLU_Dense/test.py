###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""test FusedMaxPoolConv1DReLU -> Dense
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
test_input = np.random.normal(0, 0.5, size=(7, 7))

print(test_input.shape)
test_input = clamp(np.floor(test_input*128+0.5))/128.0
print(test_input.shape)
test_input = np.reshape(test_input, (1, 7, 7))
print('Test Input shape', test_input.shape)
print('Test Input', test_input)

# Init layer kernel
k_size = 7*2
init_kernel = np.linspace(-0.9, 0.9, num=k_size, dtype=np.float32)
init_kernel = clamp(np.floor(init_kernel*128+0.5))/128.0

kernel_initializer = tf.keras.initializers.constant(init_kernel)

init_bias = np.array([0.5])
bias_initializer = tf.keras.initializers.constant(init_bias)

k3_size = 6*3
init_kernel3 = np.linspace(-0.5, 0.5, num=k3_size, dtype=np.float32)
init_kernel3 = clamp(np.floor(init_kernel3*128+0.5))/128.0
kernel_initializer3 = tf.keras.initializers.constant(init_kernel3)

# Create functional model
input_layer = tf.keras.Input(shape=(7, 7))
reshape = tf.keras.layers.Reshape(target_shape=(7, 7))(input_layer)

conv1 = ai8xTF.FusedMaxPoolConv1DReLU(
    filters=2,
    kernel_size=1,
    strides=1,
    padding_size=0,
    pool_size=2,
    pool_strides=2,
    use_bias=False,
    kernel_initializer=kernel_initializer,
    )(reshape)
flat = tf.keras.layers.Flatten()(conv1)
output_layer = ai8xTF.FusedDense(
    3,
    wide=True,
    use_bias=False,
    kernel_initializer=kernel_initializer3,
    )(flat)
model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

for layer in model.layers:
    weight = np.array((layer.get_weights()[0:1]))  # weights
    # Convert to 8bit and round
    print('Weight(8-bit)=\n', clamp(np.floor(weight*128+0.5)))
    bias = (layer.get_weights()[1:2])  # bias
    print('Bias=', bias)
    tf.print(f"Layer: {layer.get_config ()['name']} \
            Wmin: {tf.math.reduce_min(weight)}, \
            Wmax: {tf.math.reduce_max(weight)}, \
            Bias min: {tf.math.reduce_min(bias)}, \
            Bias max: {tf.math.reduce_min(bias)}")


output = model.predict(test_input)

# Model output
print('Model output =', output)

# Save model
tf.saved_model.save(model, 'saved_model')

saved_input = clamp(np.floor(test_input*128+0.5))
print('Input(8-bit)\n:', saved_input)
# saved_input = saved_input.flatten()
saved_input = saved_input.swapaxes(0, 2)
# saved_input = saved_input.swapaxes(0, 1)


print(saved_input.shape)
# Save input
np.save(os.path.join(logdir, 'input_sample_7x7x1.npy'), np.array(saved_input, dtype=np.int32))
print('Output(8-bit):\n', clamp(np.floor(output*128+0.5)))
print(output.shape)
sys.exit(0)
