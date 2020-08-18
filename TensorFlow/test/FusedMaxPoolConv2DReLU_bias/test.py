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
k_size = 18
init_kernel = np.linspace(-0.9, 0.9, num=k_size, dtype=np.float32)

kernel_initializer = tf.keras.initializers.constant(init_kernel)

init_bias = np.array([-0.5, 0.5])
bias_initializer = tf.keras.initializers.constant(init_bias)

# Create functional model
input_layer = tf.keras.Input(shape=(4, 4))
reshape = tf.keras.layers.Reshape(target_shape=(4, 4, 1))(input_layer)
conv1 = ai8xTF.FusedMaxPoolConv2DReLU(
    filters=2,
    kernel_size=3,
    strides=1,
    padding_size=1,
    use_bias=True,
    kernel_initializer=kernel_initializer,
    bias_initializer=bias_initializer
    )(reshape)
flat = tf.keras.layers.Flatten()(conv1)
model = tf.keras.Model(inputs=[input_layer], outputs=[flat])


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
