#!/usr/bin/env python3
'''Module contains build_model function'''
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''Function builds a neural network with Keras library'''

    input = K.Input(shape=(nx, ))
    regularizer = K.regularizers.L2(lambtha)

    for idx in range(len(layers)):
        if idx != 0:
            dropout = K.layers.Dropout(1 - keep_prob)(layer)
            layer = K.layers.Dense(layers[idx], activation=activations[idx],
                                   kernel_regularizer=regularizer)(dropout)
        else:
            layer = K.layers.Dense(layers[idx], activation=activations[idx],
                                   kernel_regularizer=regularizer)(input)

    return K.Model(input, layer)
