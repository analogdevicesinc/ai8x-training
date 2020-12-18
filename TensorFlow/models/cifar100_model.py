###################################################################################################
#
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
cifar10 network for AI85/AI86
"""
import tensorflow as tf

# AI8xTF sub-classes
import ai8xTF  # pylint: disable=import-error

regularizer_rate = 0.00001/20
regularizer = None  # tf.keras.regularizers.l2(regularizer_rate)
activity_regularizer = tf.keras.regularizers.l2(0.00001/20)
kernel_init = 'glorot_uniform'

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),
    ai8xTF.FusedConv2DReLU(
        filters=16,
        kernel_size=3,
        strides=1,
        padding_size=1,
        use_bias=False,
        kernel_regularizer=regularizer,
        activity_regularizer=activity_regularizer,
        kernel_initializer=kernel_init),
    ai8xTF.FusedConv2DReLU(
        filters=20,
        kernel_size=3,
        strides=1,
        padding_size=1,
        use_bias=False,
        kernel_regularizer=regularizer,
        activity_regularizer=activity_regularizer,
        kernel_initializer=kernel_init),
    ai8xTF.FusedConv2DReLU(
        filters=20,
        kernel_size=3,
        strides=1,
        padding_size=1,
        use_bias=False,
        kernel_regularizer=regularizer,
        activity_regularizer=activity_regularizer,
        kernel_initializer=kernel_init),
    ai8xTF.FusedConv2DReLU(
        filters=20,
        kernel_size=3,
        strides=1,
        padding_size=1,
        use_bias=False,
        kernel_regularizer=regularizer,
        activity_regularizer=activity_regularizer,
        kernel_initializer=kernel_init),
    # tf.keras.layers.Dropout(0.2),
    ai8xTF.FusedMaxPoolConv2DReLU(
        filters=20,
        kernel_size=3,
        strides=1,
        padding_size=1,
        pool_size=2,
        pool_strides=2,
        use_bias=False,
        kernel_regularizer=regularizer,
        activity_regularizer=activity_regularizer,
        kernel_initializer=kernel_init),
    ai8xTF.FusedConv2DReLU(
        filters=20,
        kernel_size=3,
        strides=1,
        padding_size=1,
        use_bias=False,
        kernel_regularizer=regularizer,
        activity_regularizer=activity_regularizer,
        kernel_initializer=kernel_init),
    ai8xTF.FusedConv2DReLU(
        filters=44,
        kernel_size=3,
        strides=1,
        padding_size=1,
        use_bias=False,
        kernel_regularizer=regularizer,
        activity_regularizer=activity_regularizer,
        kernel_initializer=kernel_init),
    ai8xTF.FusedMaxPoolConv2DReLU(
        filters=48,
        kernel_size=3,
        strides=1,
        padding_size=1,
        pool_size=2,
        pool_strides=2,
        use_bias=False,
        kernel_regularizer=regularizer,
        activity_regularizer=activity_regularizer,
        kernel_initializer=kernel_init),
    # tf.keras.layers.Dropout(0.2),
    ai8xTF.FusedConv2DReLU(
        filters=48,
        kernel_size=3,
        strides=1,
        padding_size=1,
        use_bias=False,
        kernel_regularizer=regularizer,
        activity_regularizer=activity_regularizer,
        kernel_initializer=kernel_init),
    ai8xTF.FusedMaxPoolConv2DReLU(
        filters=96,
        kernel_size=3,
        strides=1,
        padding_size=1,
        pool_size=2,
        pool_strides=2,
        use_bias=False,
        kernel_regularizer=regularizer,
        activity_regularizer=activity_regularizer,
        kernel_initializer=kernel_init),
    ai8xTF.FusedMaxPoolConv2DReLU(
        filters=512,
        kernel_size=1,
        strides=1,
        padding_size=0,
        pool_size=2,
        pool_strides=2,
        use_bias=False,
        kernel_regularizer=regularizer,
        activity_regularizer=activity_regularizer,
        kernel_initializer=kernel_init),
    ai8xTF.FusedConv2DReLU(
        filters=128,
        kernel_size=1,
        strides=1,
        padding_size=0,
        use_bias=False,
        kernel_regularizer=regularizer,
        activity_regularizer=activity_regularizer,
        kernel_initializer=kernel_init),
    ai8xTF.FusedMaxPoolConv2DReLU(
        filters=128,
        kernel_size=3,
        strides=1,
        padding_size=1,
        pool_size=2,
        pool_strides=2,
        use_bias=False,
        kernel_regularizer=regularizer,
        activity_regularizer=activity_regularizer,
        kernel_initializer=kernel_init),
    # tf.keras.layers.Dropout(0.2),
    ai8xTF.Conv2D(
        filters=100,
        kernel_size=1,
        strides=1,
        padding_size=0,
        wide=True,
        use_bias=False,
        kernel_regularizer=regularizer,
        activity_regularizer=activity_regularizer,
        kernel_initializer=kernel_init),
    tf.keras.layers.Flatten()
])

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    mode='max',
    factor=0.2,
    patience=3,
    verbose=1,
    min_lr=1e-5)
