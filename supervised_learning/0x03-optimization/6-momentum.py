#!/usr/bin/env python3
'''Module contains create_momentum_op function'''
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    '''
    Function creates the training operation for a neural network in tensorflow
    using the gradient descent with momentum optimization algorithm
    '''
    momentum_op = tf.train.MomentumOptimizer(alpha, beta1)
    return momentum_op.minimize(loss)
