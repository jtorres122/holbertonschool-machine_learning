#!/usr/bin/env python3
'''Module contains save and load weight fucntions'''
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    '''Function saves a model's weight'''
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    '''Function loads a model's weight'''
    network.load_weights(filename)
    return None
