#!/usr/bin/env python3
'''Module contains create_batch_norm_layer function'''
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    '''Function'''
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=init)
    return layer(prev)
