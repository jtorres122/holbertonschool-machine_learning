#!/usr/bin/env python3
'''Module contains build_model function'''
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''
    Function builds a neural network with the Keras library
    using the sequential class
    '''
    model = K.Sequential()
    regularizer = K.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=regularizer,
                                     input_shape=(nx,)))
        else:
            model.add(K.layers.Dropout(1 - keep_prob))
            model.add(K.layers.Dense(layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=regularizer))
    return model
