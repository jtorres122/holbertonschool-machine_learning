#!/usr/bin/env python3
'''Module contains the calculate_loss function'''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def calculate_loss(y, y_pred):
    '''calculates the softmax cross-entropy loss of a prediction'''
    return tf.losses.softmax_cross_entropy(y, y_pred)
