#!/usr/bin/env python3
'''Module contains build_model function'''
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''Function builds a neural network with the Keras library'''
    model = K.Sequential()
    for i in range(len(layers)):
        model.add(K.layers.Dense(layers[i],
                                 activation=activations[i],
                                 kernel_regularizer=K.regularizers.l2(lambtha),
                                 input_shape=(nx,)))
                                 
        model.add(K.layers.Dropout(1 - keep_prob))
    return model
