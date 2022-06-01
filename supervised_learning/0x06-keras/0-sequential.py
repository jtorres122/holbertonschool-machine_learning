#!/usr/bin/env python3
'''Module contains build_model function'''
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''Function builds a neural network with the Keras library'''
    model = K.Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=K.regularizers.l2(lambtha),
                                     input_shape=(nx,)))
        else:
            model.add(K.layers.Dropout(1 - keep_prob))
            model.add(K.layers.Dense(layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=K.regularizers.l2(lambtha)))
    return model
