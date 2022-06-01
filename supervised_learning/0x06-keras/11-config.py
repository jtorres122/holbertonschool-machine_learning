#!/usr/bin/env python3
'''Module contains save_config and load_config functions'''
import tensorflow.keras as K


def save_config(network, filename):
    '''
    Function saves a model's configuration in JSON format:
    network is the model whose configuration should be saved
    filename is the path of the file that the configuration should be saved to
    Returns: None
    '''
    json_string = network.to_json()
    open(filename, 'w').write(json_string)
    return None


def load_config(filename):
    '''
    Function loads a model with a specific configuration:
    filename is the path of the file containing the model's
        configuration in JSON format
    Returns: the loaded model
    '''
    return K.models.model_from_json(open(filename).read())
