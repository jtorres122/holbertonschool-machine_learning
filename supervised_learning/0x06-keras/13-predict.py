#!/usr/bin/env python3
'''Module contains predict function '''
import tensorflow.keras as K


def predict(network, data, verbose=False):
    '''Function makes a prediction using a neural network'''

    return network.predict(data, verbose=verbose)
