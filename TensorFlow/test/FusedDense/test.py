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
test_input = np.random.normal(0, 0.5, size=10)
test_input = test_input.reshape(1, 10)
print ('Test Input shape', test_input.shape)
print('Test Input', test_input)

# Init layer kernel
k_size = 70
init_kernel = np.linspace(-0.9, 0.9, num=k_size, dtype=np.float32)
#init_kernel = np.random.normal(0, 0.5, size=k_size)
kernel_initializer = tf.keras.initializers.constant(init_kernel)

init_bias = np.array([0.5])
bias_initializer = tf.keras.initializers.constant(init_bias)


# Create functional model
input_layer = tf.keras.Input(shape=(10))
reshape = tf.keras.layers.Reshape(target_shape=(10, 1))(input_layer)
flat = tf.keras.layers.Flatten()(reshape)
output_layer = ai8xTF.FusedDense(7, wide=True, kernel_initializer=kernel_initializer)(flat)

model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])


model.compile( optimizer = 'adam' ,
                loss = tf.keras.losses.SparseCategoricalCrossentropy ( from_logits = True ),
                metrics = ['accuracy'])

print('Shape flat', flat.shape)

model.summary()

for layer in model.layers:
      weight = np.array((layer.get_weights()[0:1])) #weights
      # Convert to 8bit and round
      print('Weight(8bit)=\n', np.floor(weight*128+0.5))
      bias = (layer.get_weights()[1:2]) #bias
      print('Bias=', bias)
      tf.print(f"Layer: {layer.get_config ()['name']} \
                Wmin: {tf.math.reduce_min(weight)}, \
                Wmax: {tf.math.reduce_max(weight)}, \
                Bias min: {tf.math.reduce_min(bias)}, \
                Bias max: {tf.math.reduce_min(bias)}")


output = model.predict(test_input)

# Model output
print('Output shape =', output.shape)
print('Model output =', output)

# Save model
tf.saved_model.save(model,'saved_model')
# Convert to 8bit and round
saved_input = np.floor(test_input*128+0.5)
print('Input(8-bit):', saved_input)
# Save input
#np.save (os.path.join(logdir, 'input_sample_1x10.npy'), np.array(saved_input, dtype=np.int32))
saved_input1 = saved_input.swapaxes(0,1)
np.save (os.path.join(logdir, 'input_sample_10x1.npy'), np.array(saved_input1, dtype=np.int32))
# Convert to 8bit and round
print('Output(8-bit):', np.floor(output*128+0.5))

exit(0)
