#!/usr/bin/env python3
'''Module contains the evaluate function'''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def evaluate(X, Y, save_path):
    '''Function evaluates the output of a NN'''
    model = tf.keras.models.load_model(save_path)
    return model.evaluate(X, Y)
