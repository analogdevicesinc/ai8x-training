import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import ai8xTF
from random import randint
import os
import sys

ai8xTF.set_device (85 , False , 10 )

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

# following piece it to init seed to make repeated results
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(10)
tf.random.set_seed(7)

logdir = 'saved_model'

if not os.path.isdir(logdir):
    os.makedirs(logdir)

# Log stdout to file
sys.stdout = Logger(os.path.join(logdir, 'result.log'))

# Init input samples
test_input = np.random.normal(0, 0.5, size=(4, 4))
test_input = test_input.reshape(1, 4, 4)
print ('Test Input shape', test_input.shape)
print('Test Input', test_input)

# Init layer kernel

k1 = np.linspace(-0.9, 0.9, num=18, dtype=np.float32)
k2 = np.linspace(-0.7, 0.7, num=18, dtype=np.float32)
k3 = np.linspace(-0.5, 0.5, num=9, dtype=np.float32)
k4 = np.linspace(-0.3, 0.3, num=9, dtype=np.float32)
k5 = np.linspace(-0.1, 0.1, num=5, dtype=np.float32)

init_bias = np.array([-0.5, 0.5])
bias_initializer = tf.keras.initializers.constant(init_bias)

# Create functional model
input_layer = tf.keras.Input(shape=(4, 4))
reshape = tf.keras.layers.Reshape(target_shape=(4, 4, 1))(input_layer)

conv1 = ai8xTF.FusedConv2DReLU(
    filters=2,
    kernel_size=3,
    strides=1,
    padding_size=1,
    use_bias=False,
    kernel_initializer=tf.keras.initializers.constant(k1)
    )(reshape)

conv2 = ai8xTF.FusedMaxPoolConv2DReLU(
    filters=1,
    kernel_size=3,
    strides=1,
    padding_size=2,
    pool_size=2,
    pool_strides=2,
    use_bias=False,
    kernel_initializer = tf.keras.initializers.constant(k2)
    )(conv1)

conv3 = ai8xTF.FusedMaxPoolConv2DReLU(
    filters=1,
    kernel_size=3,
    strides=1,
    padding_size=1,
    pool_size=2,
    pool_strides=2,
    use_bias=False,
    kernel_initializer=tf.keras.initializers.constant(k3)
    )(conv2)

conv4 = ai8xTF.FusedAvgPoolConv2DReLU(
    filters=1,
    kernel_size=3,
    strides=1,
    padding_size=1,
    pool_size=2,
    pool_strides=2,
    use_bias=False,
    kernel_initializer=tf.keras.initializers.constant(k4)
    )(conv3)

flat = tf.keras.layers.Flatten()(conv4)

output_layer = ai8xTF.FusedDense(5, wide=True, kernel_initializer=tf.keras.initializers.constant(k5))(flat)

model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])

model.compile( optimizer = 'adam' ,
                loss = tf.keras.losses.SparseCategoricalCrossentropy ( from_logits = True ),
                metrics = ['accuracy'])

model.summary()

for layer in model.layers:
      weight = (layer.get_weights()[0:1]) #weights
      print('Weight=', weight)
      bias = (layer.get_weights()[1:2]) #bias
      print('Bias=', bias)
      tf.print(f"Layer: {layer.get_config ()['name']} \
                Wmin: {tf.math.reduce_min(weight)}, \
                Wmax: {tf.math.reduce_max(weight)}, \
                Bias min: {tf.math.reduce_min(bias)}, \
                Bias max: {tf.math.reduce_min(bias)}")

output = model.predict(test_input)

# Model output
print('Output=', output)

# Save model
tf.saved_model.save(model,'saved_model')

saved_input = np.trunc(test_input * 128)
print('Save Input as int8:', saved_input)
# Save input
np.save (os.path.join(logdir, 'input_sample_1x4x4.npy'), np.array(saved_input, dtype=np.int32))
print('Output (int):', np.trunc(output*127))

exit(0)
