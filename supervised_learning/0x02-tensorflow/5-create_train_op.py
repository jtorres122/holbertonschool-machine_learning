#!/usr/bin/env python3
'''Module contains the create_train_op function'''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_train_op(loss, alpha):
    '''Function creates training operation for the NN'''
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
