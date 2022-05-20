#!/usr/bin/env python3
'''Module contains learning_rate_decay function'''
import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    '''Function  updates the learning rate using inverse time decay in numpy'''
    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)
