#!/usr/bin/env python3
'''Module contains the batch_norm function'''
import tensorflow.compat.v1 as tf


def batch_norm(Z, beta, gamma, epsilon):
    '''
    Function normalizes an unactivated output 
    of a neural network using batch normalization
    '''
    batch_mean, batch_var = tf.nn.moments(Z, [0])
    return tf.nn.batch_normalization(Z, batch_mean, batch_var,
                                     beta, gamma, epsilon)
