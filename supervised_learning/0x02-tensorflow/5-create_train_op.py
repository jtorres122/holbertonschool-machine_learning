#!/usr/bin/env python3
'''Module contains create_train_op function'''
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    '''Creates the training operation'''
    train_op = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return train_op
