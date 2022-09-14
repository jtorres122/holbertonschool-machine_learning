#!/usr/bin/env python3
'''Module contains optimize_model function'''
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    '''Function trains a model using mini-batch gradient descent'''

    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle)
