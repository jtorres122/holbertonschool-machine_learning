#!/usr/bin/env python3
'''Module contains create_RMSProp_op function'''
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    '''Function creates the RMSProp optimization operation'''
    opt = tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon)
    return opt.minimize(loss)
