#!/usr/bin/env python3
'''Module contains create_RMSProp_op function'''
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    '''Function creates the RMSProp optimization operation'''
    s = tf.Variable(tf.zeros(loss.shape), trainable=False)
    opt = tf.train.RMSPropOptimizer(alpha, beta2, epsilon, s)
    return opt.minimize(loss)
