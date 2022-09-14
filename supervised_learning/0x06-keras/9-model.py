#!/usr/bin/env python3
'''Module contains save and load model functions'''
import tensorflow.keras as K


def save_model(network, filename):
    '''Function saves an entire model'''
    network.save(filename)
    return None


def load_model(filename):
    '''Function loads an entire model'''
    return K.models.load_model(filename)
