#!/usr/bin/env python3
'''Module contains build_model function'''
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''Function builds a neural network with Keras library'''

    model = K.Sequential()
    regularizer = K.regularizers.L2(lambtha)

    for idx in range(len(layers)):
        if idx != 0:
            model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(layers[idx], input_shape=(nx, ),
                  activation=activations[idx], kernel_regularizer=regularizer))

    return model
