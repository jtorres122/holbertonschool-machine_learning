#!/usr/bin/env python3
'''Module contains the evaluate function'''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def evaluate(X, Y, save_path):
    '''Function evaluates the output of a NN'''
    return tf.keras.model.evaluate(X, Y, save_path)
