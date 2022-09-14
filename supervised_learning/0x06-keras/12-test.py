#!/usr/bin/env python3
'''Module contains test_model function '''
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    '''Function tests a neural network'''

    return network.evaluate(data, labels, verbose=verbose)
