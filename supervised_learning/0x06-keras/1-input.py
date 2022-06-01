#!/usr/bin/env python3
'''Module contains build_model function'''
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''
    Function builds a neural network with the Keras library
    using the input class
    '''
    inputs = K.Input(shape=(nx,))
    regularizer = K.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            outputs = (K.layers.Dense(units=layers[i],
                                      activation=activations[i],
                                      kernel_regularizer=regularizer)(inputs))
        else:
            outputs = K.layers.Dropout(1 - keep_prob)(outputs)
            outputs = K.layers.Dense(layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=regularizer)(outputs)

    model = K.Model(inputs=inputs, outputs=outputs)
    return model
