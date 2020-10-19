###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""test FusedAvgPoolConv2DReLU with Bias
"""
import os
import sys
import tensorflow as tf
import numpy as np
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
test_input = np.random.normal(0, 0.5, size=(4, 4))
print(test_input.shape)
test_input = clamp(np.floor(test_input*128 + 0.5))/128.0
# print (test_input.shape)
test_input = np.reshape(test_input, (1, 4, 4))
print('Test Input shape', test_input.shape)
print('Test Input', test_input)

# Init layer kernel
k_size = 18
init_kernel = np.linspace(-0.9, 0.9, num=k_size, dtype=np.float32)
init_kernel = clamp(np.floor(init_kernel*128+0.5))/128.0

kernel_initializer = tf.keras.initializers.constant(init_kernel)

init_bias = np.array([-0.4, 0.4])
init_bias = clamp(np.floor(init_bias*128+0.5))/128.0
bias_initializer = tf.keras.initializers.constant(init_bias)

# Create functional model
input_layer = tf.keras.Input(shape=(4, 4))
reshape = tf.keras.layers.Reshape(target_shape=(4, 4, 1))(input_layer)
conv1 = ai8xTF.FusedAvgPoolConv2DReLU(
    filters=2,
    kernel_size=3,
    strides=1,
    padding_size=1,
    use_bias=True,
    kernel_initializer=kernel_initializer,
    bias_initializer=bias_initializer
    )(reshape)
# flat = tf.keras.layers.Flatten()(conv1)
model = tf.keras.Model(inputs=[input_layer], outputs=[conv1])

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
    print(bias.shape)
    tf.print(f"Layer: {layer.get_config ()['name']} \
              Wmin: {tf.math.reduce_min(weight)}, \
              Wmax: {tf.math.reduce_max(weight)}, \
              Bias min: {tf.math.reduce_min(bias)}, \
              Bias max: {tf.math.reduce_min(bias)}")


output = model.predict(test_input)

# Model output
print('Model output:\n', output)

# Save model
tf.saved_model.save(model, 'saved_model')

# Convert to 8bit, round and clamp
saved_input = clamp(np.floor(test_input*128+0.5))
print('Input(8-bit):\n', saved_input)
print(saved_input.shape)
# Save input
np.save(os.path.join(logdir, 'input_sample_1x4x4.npy'), np.array(saved_input, dtype=np.int32))
# Convert to 8bit, round and clamp
print('Output(8-bit):\n', clamp(np.floor(output*128+0.5)))
print(output.shape)

sys.exit(0)
