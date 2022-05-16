#!/usr/bin/env python3
'''Module contains create_layer function'''
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    '''Creates a layer of a neural network'''
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n, activation=activation, kernel_initializer=init)
    return layer(prev)
