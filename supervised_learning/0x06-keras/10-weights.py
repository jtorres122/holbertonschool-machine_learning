#!/usr/bin/env python3
'''Module contains save_model and load_model functions'''
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    '''
    Function saves a model's weights:
    network is the model whose weights should be saved
    filename is the path of the file that the weights should be saved to
    save_format is the format in which the weights should be saved
    Returns: None
    '''
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    '''
    Function loads a model's weights:
    network is the model to which the weights should be loaded
    filename is the path of the file that the weights should be loaded from
    Returns: None
    '''
    network.load_weights(filename)
    return None
