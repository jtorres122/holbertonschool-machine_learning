#!/usr/bin/env python3
'''Module contains create_layer function'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_layer(prev, n, activation):
    '''Function creates a tensor layer'''
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(
        name="layer", units=n, activation=activation,
        kernel_initializer=initializer)
    y = layer(prev)
    return y
