#!/usr/bin/env python3
'''Module contains forward_prop function'''
import tensorflow.compat.v1 as tf


create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    '''Creates a forward propagation graph'''
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    for i in range(len(layer_sizes)):
        layer = tf.layers.Dense(units=layer_sizes[i],
                                activation=activations[i],
                                kernel_initializer=init)
        prev = layer(x)
    return prev
