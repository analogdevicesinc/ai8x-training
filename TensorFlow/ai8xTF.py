###################################################################################################
#
# Copyright (C) 2019-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Contains the limits of the AI84/AI85/AI87 implementations and custom Tensorflow modules that take
the limits into account.
"""
import tensorflow as tf
# import tensorflow.keras as keras
from tensorflow import keras

dev = None


# --------------------------- Conv1D
class Conv1D(keras.layers.Layer):  # pylint: disable=too-many-instance-attributes
    """
    AI8X - Fused 1D pooling ('Avg', 'Max' or None) followed by
    1D convolution and activation ('relu', None)
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding_size=0,
                 pooling=None,
                 pool_size=2,
                 pool_strides=2,
                 activation=None,
                 use_bias=True,
                 output_shift=0,
                 wide=False,
                 **kwargs):
        super().__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding_size = padding_size
        self.pooling = pooling
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.activation = activation
        self.use_bias = use_bias
        self.output_shift = output_shift
        self.wide = wide

        assert self.filters <= 1024
        assert not wide or activation is None

        if pooling is not None:
            if pool_strides is None:
                pool_strides = pool_size

            assert dev.device != 84 or pool_size & 1 == 0
            assert pool_size <= 16 and (dev.device != 84 or pool_size <= 4
                                        or pooling == 'Max')
            assert 0 < pool_strides <= 16 and (dev.device != 84
                                               or pool_strides <= 4
                                               or pooling == 'Max')
            assert strides == 1
        else:
            assert dev.device != 84 or strides == 3
            assert dev.device == 84 or strides == 1

        if pooling == 'Max':
            self.pool = keras.layers.MaxPool1D(
                pool_size=pool_size, strides=pool_strides, padding='valid')
        elif pooling == 'Avg':
            self.pool = keras.layers.AveragePooling1D(
                pool_size=pool_size, strides=pool_strides, padding='valid')
        else:
            self.pool = None

        self.zeropadding = keras.layers.ZeroPadding1D(padding=padding_size)

        if kernel_size is not None:
            assert dev.device != 84 or padding_size in [0, 3, 6]
            assert dev.device == 84 or padding_size in [0, 1, 2]
            assert dev.device != 84 or kernel_size == 9
            assert dev.device == 84 or kernel_size in [
                1, 2, 3, 4, 5, 6, 7, 8, 9
            ]

            self.conv1d = keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='valid',
                activation=activation,
                use_bias=use_bias,
                **kwargs)
        else:
            self.conv1d = None

        self.quantize_pool, self.clamp_pool = quantize_clamp_pool(pooling)
        self.quantize, self.clamp = quantize_clamp(wide, output_shift)

    def call(self, x):  # pylint: disable=arguments-differ
        if self.pool is not None:
            x = self.clamp_pool(self.pool(x))
            # print('PoolOutput:%s' % (x.shape))
        if self.conv1d is not None:
            x = self.zeropadding(x)
            x = self.conv1d(x)
            x = self.clamp(x)
            # print('Conv&PaddingOutput:%s' % (x.shape))

        return x

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding_size": self.padding_size,
            "pooling": self.pooling,
            "pool_size": self.pool_size,
            "pool_strides": self.pool_strides,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "output_shift": self.output_shift,
            "wide": self.wide
        }


class FusedConv1D(Conv1D):
    """
    AI8X - Fused 1D Convolution, Activation = None
    """

    def __init__(self, *args, **kwargs):
        check_if_pooling_arg(**kwargs)
        super().__init__(*args, activation=None, **kwargs)


class FusedConv1DReLU(Conv1D):
    """
    AI8X - Fused 1D Convolution and ReLU
    """

    def __init__(self, *args, **kwargs):
        check_if_pooling_arg(**kwargs)
        super().__init__(
            *args, activation='relu', **kwargs)


class FusedMaxPoolConv1D(Conv1D):
    """
    AI8X - Fused 1D Max Pool, 1D Convolution and Activation = None
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, pooling='Max', **kwargs)


class FusedMaxPoolConv1DReLU(Conv1D):
    """
    AI8X - Fused 1D Max Pool, 1D Convolution and Activation = 'relu''
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, pooling='Max', activation='relu', **kwargs)


class MaxPool1D(Conv1D):
    """
    AI8X - 1D Max Pool
    """

    def __init__(self, pool_size, pool_strides=None, **kwargs):
        check_if_conv_arg(**kwargs)
        super().__init__(
            0,
            None,
            pool_size=pool_size,
            pool_strides=pool_strides,
            pooling='Max',
            activation=None,
            **kwargs)


class FusedAvgPoolConv1D(Conv1D):
    """
    AI8X - Fused 1D Avg Pool, 1D Convolution and Activation = None
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, pooling='Avg', **kwargs)


class FusedAvgPoolConv1DReLU(Conv1D):
    """
    AI8X - Fused 1D Avg Pool, 1D Convolution and Activation = 'relu''
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, activation='relu', pooling='Avg', **kwargs)


class AvgPool1D(Conv1D):
    """
    AI8X - 1D Avg Pool
    """

    def __init__(self, pool_size, pool_strides=None, **kwargs):
        check_if_conv_arg(**kwargs)
        super().__init__(
            0,
            None,
            pool_size=pool_size,
            pool_strides=pool_strides,
            pooling='Avg',
            activation=None,
            **kwargs)


# -------------------------- Conv2D


class Conv2D(keras.layers.Layer):  # pylint: disable=too-many-instance-attributes
    """
    AI8X - 2D pooling ('Avg', 'Max' or None) optionally followed by
    2D convolution/transposed 2D convolution and activation ('relu', None)
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding_size=0,
                 op='Conv2d',
                 pooling=None,
                 pool_size=2,
                 pool_strides=2,
                 activation=None,
                 use_bias=True,
                 output_shift=0,
                 wide=False,
                 **kwargs):
        super().__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding_size = padding_size
        self.pooling = pooling
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.activation = activation
        self.use_bias = use_bias
        self.output_shift = output_shift
        self.wide = wide
        self.op = op

        assert self.filters <= 1024
        assert not wide or activation is None

        if pooling is not None:
            if pool_strides is None:
                pool_strides = pool_size

            if isinstance(pool_size, int):
                assert dev.device != 84 or pool_size & 1 == 0
                assert pool_size <= 16 and (dev.device != 84 or pool_size <= 4
                                            or pooling == 'Max')
            elif isinstance(pool_size, tuple):
                assert len(pool_size) == 2
                assert dev.device != 84 or pool_size[0] & 1 == 0
                assert pool_size[0] <= 16 and (dev.device != 84
                                               or pool_size[0] <= 4
                                               or pooling == 'Max')
                assert dev.device != 84 or pool_size[1] & 1 == 0
                assert pool_size[1] <= 16 and (dev.device != 84
                                               or pool_size[1] <= 4
                                               or pooling == 'Max')
            else:
                raise ValueError('pool_size must be int or tuple')

            if isinstance(pool_strides, int):
                assert pool_strides > 0
                assert pool_strides <= 16 and (dev.device != 84
                                               or pool_strides <= 4
                                               or pooling == 'Max')
            elif isinstance(pool_strides, tuple):
                assert len(pool_strides) == 2
                assert dev.device != 84 or pool_strides[0] == pool_strides[1]
                assert 0 < pool_strides[0] <= 16 and (dev.device != 84
                                                      or pool_strides[0] <= 4
                                                      or pooling == 'Max')
                assert 0 < pool_strides[1] <= 16 and (dev.device != 84
                                                      or pool_strides[1] <= 4
                                                      or pooling == 'Max')
            else:
                raise ValueError('pool_strides must be int or tuple')

            assert strides == 1
        else:
            if op == 'Conv2d':
                if pool_strides > 0:
                    # assert strides == 16
                    pass
                else:
                    assert strides == 1
            elif op == 'ConvTranspose2d':
                assert strides == 2

        assert 0 <= padding_size <= 2

        if pooling == 'Max':
            self.pool = keras.layers.MaxPool2D(
                pool_size=pool_size, strides=pool_strides, padding='valid')
        elif pooling == 'Avg':
            self.pool = keras.layers.AveragePooling2D(
                pool_size=pool_size, strides=pool_strides, padding='valid')
        else:
            self.pool = None

        self.zeropadding = keras.layers.ZeroPadding2D(padding=padding_size)

        if kernel_size is not None:
            if isinstance(kernel_size, tuple):
                assert len(
                    kernel_size) == 2 and kernel_size[0] == kernel_size[1]
                kernel_size = kernel_size[0]

            assert kernel_size == 3 or dev.device != 84 and kernel_size == 1

            if op == 'Conv2d':
                self.conv2d = keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='valid',
                    activation=activation,
                    use_bias=use_bias,
                    **kwargs)

            elif op == 'ConvTranspose2d':
                assert dev.device != 84
                self.conv2d = keras.layers.Conv2DTranspose(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='same',
                    output_padding=1,
                    activation=activation,
                    use_bias=use_bias,
                    **kwargs)
            else:
                raise ValueError('Unsupported operation')
        else:
            self.conv2d = None

        self.quantize_pool, self.clamp_pool = quantize_clamp_pool(pooling)
        self.quantize, self.clamp = quantize_clamp(wide, output_shift)

    def call(self, x):  # pylint: disable=arguments-differ
        if self.pool is not None:
            x = self.clamp_pool(self.pool(x))
            # print('PoolOutput:%s'%(x.shape))
        if self.conv2d is not None:
            x = self.zeropadding(x)
            x = self.conv2d(x)
            x = self.clamp(x)
            # print('Conv&PaddingOutput:%s'%(x.shape))

        return x

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            # "padding_size": self.padding_size,
            # "pooling": self.pooling,
            # "pool_size": self.pool_size,
            # "pool_strides": self.pool_strides,
            "activation": self.activation,
            "use_bias": self.use_bias,
            # "output_shift": self.output_shift,
            # "wide": self.wide,
            # "op": self.op
        }


class FusedConv2D(Conv2D):
    """
    AI8X - Fused 2D Convolution, Activation = None
    """

    def __init__(self, *args, **kwargs):
        check_if_pooling_arg(**kwargs)
        super().__init__(*args, activation=None, **kwargs)


class FusedConv2DReLU(Conv2D):
    """
    AI8X - Fused 2D Convolution and ReLU
    """

    def __init__(self, *args, **kwargs):
        check_if_pooling_arg(**kwargs)
        super().__init__(
            *args, activation='relu', **kwargs)


class FusedMaxPoolConv2D(Conv2D):
    """
    AI8X - Fused 2D Max Pool, 2D Convolution and Activation = None
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, pooling='Max', **kwargs)


class FusedMaxPoolConv2DReLU(Conv2D):
    """
    AI8X - Fused 2D Max Pool, 2D Convolution and Activation = 'relu''
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, activation='relu', pooling='Max', **kwargs)


class MaxPool2D(Conv2D):
    """
    AI8X - 2D Max Pool
    """

    def __init__(self, pool_size, pool_strides=None, **kwargs):
        check_if_conv_arg(**kwargs)
        super().__init__(
            0,
            None,
            pool_size=pool_size,
            pool_strides=pool_strides,
            pooling='Max',
            activation=None,
            **kwargs)


class FusedAvgPoolConv2D(Conv2D):
    """
    AI8X - Fused 2D Avg Pool, 2D Convolution and Activation = None
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, pooling='Avg', **kwargs)


class FusedAvgPoolConv2DReLU(Conv2D):
    """
    AI8X - Fused 2D Avg Pool, 2D Convolution and Activation = 'relu''
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, activation='relu', pooling='Avg', **kwargs)


class AvgPool2D(Conv2D):
    """
    AI8X - 2D Avg Pool
    """

    def __init__(self, pool_size, pool_strides=None, **kwargs):
        check_if_conv_arg(**kwargs)
        super().__init__(
            0,
            None,
            pool_size=pool_size,
            pool_strides=pool_strides,
            pooling='Avg',
            activation=None,
            **kwargs)


# -----------------------------    Conv2D Transpose


class FusedConv2DTranspose(Conv2D):
    """
    AI8X - Fused 2D Convolution Transpose, Activation = None
    """

    def __init__(self, *args, **kwargs):
        check_if_pooling_arg(**kwargs)
        super().__init__(
            *args, activation=None, op='ConvTranspose2d', **kwargs)


class FusedConv2DTransposeReLU(Conv2D):
    """
    AI8X - Fused 2D Convolution Transpose and ReLU
    """

    def __init__(self, *args, **kwargs):
        check_if_pooling_arg(**kwargs)
        super().__init__(
            *args, activation='relu', op='ConvTranspose2d', **kwargs)


class FusedMaxPoolConv2DTranspose(Conv2D):
    """
    AI8X - Fused 2D Max Pool, 2D Convolution Transpose and Activation = None
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, pooling='Max', op='ConvTranspose2d', **kwargs)


class FusedMaxPoolConv2DTransposeReLU(Conv2D):
    """
    AI8X - Fused 2D Max Pool, 2D Convolution Transpose and Activation = 'relu''
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            activation='relu',
            pooling='Max',
            op='ConvTranspose2d',
            **kwargs)


class FusedAvgPoolConv2DTranspose(Conv2D):
    """
    AI8X - Fused 2D Avg Pool, 2D Convolution Transpose and Activation = None
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, pooling='Avg', op='ConvTranspose2d', **kwargs)


class FusedAvgPoolConv2DTransposeReLU(Conv2D):
    """
    AI8X - Fused 2D Avg Pool, 2D Convolution Transpose and Activation = 'relu''
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            activation='relu',
            op='ConvTranspose2d',
            pooling='Avg',
            **kwargs)


# --------------------- Error checking


def check_if_pooling_arg(**kwargs):
    """
    AI8X - check if no pooling argument exists if the layer has pooling
    """
    for key, _ in kwargs.items():
        if key in {'pool_size', 'pool_stirdes'}:
            raise ValueError('Pooling arg without pooling')


def check_if_conv_arg(**kwargs):
    """
    AI8X - Check if conv arg is used in a pooling layer
    """
    for key, _ in kwargs.items():
        if key in {'kernel_size', 'strides', 'padding_size'}:
            raise ValueError('Conv arg without Conv layer')


# ---------------------------------   Dense Layer
class Dense(keras.layers.Layer):
    """
    AI85+ - Fused Dense and activation ('ReLU', 'Abs', None)
    """

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 output_shift=0,
                 wide=False,
                 **kwargs):
        super().__init__()

        assert units <= 1024  # output space dimension
        assert not wide or activation is None

        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.output_shift = output_shift
        self.wide = wide

        self.Dense = keras.layers.Dense(units, activation, use_bias, **kwargs)

        self.quantize, self.clamp = quantize_clamp(wide, output_shift)

    def call(self, x):  # pylint: disable=arguments-differ
        x = self.Dense(x)
        x = self.clamp(x)
        # print(x.shape)
        # print(tf.math.reduce_max(x, axis=0))
        return x

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "units": self.units,
            "activation": self.activation,
            "use_bias": self.use_bias,
            # "output_shift": self.output_shift,
            # "wide": self.wide
        }


class Clamp(keras.layers.Layer):
    """
    Post-Activation Clamping Module
    Clamp the output to the given range (typically, [-128, +127])
    """

    def __init__(self, min_val=None, max_val=None):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def call(self, x, training=None):  # pylint: disable=unused-argument, arguments-differ
        return tf.clip_by_value(x, self.min_val, self.max_val)

    def get_config(self):
        return {
            "name": self.__class__.__name__,
            "min_val": self.min_val,
            "max_val": self.max_val
            }


def quantize_clamp(wide, output_shift):  # pylint: disable=unused-argument
    """
    Return new Quantization and Clamp objects.
    """
    if dev.simulate:  # pylint: disable=no-else-raise
        raise ValueError('TODO: SUPPORT')
    else:
        quantize = Empty()
        if not wide:
            clamp = Clamp(  # Do not combine with ReLU
                min_val=-1.,
                max_val=1.,
            )
        else:
            clamp = Clamp(
                min_val=-(2.**((dev.FULL_ACC_BITS-2*(dev.DATA_BITS-1))-1)),
                max_val=2.**((dev.FULL_ACC_BITS-2*(dev.DATA_BITS-1))-1),
            )

    return quantize, clamp


def quantize_clamp_pool(pooling):
    """
    Return new Quantization and Clamp objects for pooling.
    """
    if dev.simulate:  # pylint: disable=no-else-raise
        raise ValueError('TODO: SUPPORT')
    else:
        quantize = Empty()
        if pooling == 'Avg':
            clamp = Clamp(min_val=-1., max_val=1.)
        else:  # Max, None
            clamp = Empty()

    return quantize, clamp


class FusedDenseReLU(Dense):
    """
    AI85+ - Fused Dense and ReLU
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, activation='relu', **kwargs)


class FusedDense(Dense):
    """
    AI85+ - Fused Dense
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, activation=None, **kwargs)


class Empty(keras.layers.Layer):
    """
    Do nothing
    """

    def call(self, x):  # pylint: disable=arguments-differ
        return x


# ------------------------------- Device Selection
class Device:
    """
    Device base class
    """

    def __init__(self, device, simulate, round_avg):
        self.device = device
        self.simulate = simulate
        self.round_avg = round_avg

    def __str__(self):
        return self.__class__.__name__


class DevAI84(Device):
    """
    Implementation limits for AI84
    """

    def __init__(self, simulate, round_avg):
        assert not round_avg
        super().__init__(84, simulate, round_avg)

        self.WEIGHT_BITS = 8
        self.DATA_BITS = 8
        self.ACTIVATION_BITS = 8
        self.FULL_ACC_BITS = 8
        self.FC_ACTIVATION_BITS = 16

        self.WEIGHT_INPUTS = 64
        self.WEIGHT_DEPTH = 128

        self.MAX_AVG_POOL = 4

    def __str__(self):
        return self.__class__.__name__


class DevAI85(Device):
    """
    Implementation limits for AI85
    """

    def __init__(self, simulate, round_avg):
        super().__init__(85, simulate, round_avg)

        self.WEIGHT_BITS = 8
        self.DATA_BITS = 8
        self.ACTIVATION_BITS = 8
        self.FULL_ACC_BITS = 32
        self.FC_ACTIVATION_BITS = 16

        self.WEIGHT_INPUTS = 256
        self.WEIGHT_DEPTH = 768

        self.MAX_AVG_POOL = 16

    def __str__(self):
        return self.__class__.__name__


class DevAI87(DevAI85):
    """
    Implementation limits for AI87. For now, the same as AI85.
    """

    def __str__(self):
        return self.__class__.__name__


def set_device(
        device,
        simulate,
        round_avg,
):
    """
    Change implementation configuration to match the AI84 or AI85, depending on the `device`
    integer input value and `simulate` bool. `round_avg` (AI85+) controls the average pooling
    rounding.
    """
    global dev  # pylint: disable=global-statement

    print(f'Configuring device: AI{device}, simulate={simulate}.')

    if device == 84:
        dev = DevAI84(simulate, round_avg)
    elif device == 85:
        dev = DevAI85(simulate, round_avg)
    elif device == 87:
        dev = DevAI87(simulate, round_avg)
    else:
        raise ValueError(f'Unkown device {device}.')


# set the default
set_device(85, False, 10)
