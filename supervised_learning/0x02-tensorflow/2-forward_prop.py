#!/usr/bin/env python3
'''Module contains forward_prop function'''
import tensorflow.compat.v1 as tf


create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    '''Creates a forward propagation graph'''
    for i in range(0, len(layer_sizes)):
        layer = create_layer(x, layer_sizes[i], activations[i])
    return layer
