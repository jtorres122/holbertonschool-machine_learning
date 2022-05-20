#!/usr/bin/env python3
'''Module containing the create_Adam_op function'''
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    '''
    Function that creates training operation for a NN
    using the Adam optimization operation
    '''
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                       beta2=beta2, epsilon=epsilon)
    return optimizer.minimize(loss)
