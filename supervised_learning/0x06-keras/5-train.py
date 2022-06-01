#!/usr/bin/env python3
'''Module contains train_model function'''
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    '''
    Function trains a model using mini-batch gradient descent
    while analyzing validation data
    '''
    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data)
