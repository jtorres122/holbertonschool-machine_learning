#!/usr/bin/env python3
'''Module contains l2_reg_cost function'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def l2_reg_cost(cost):
    '''Function calculates the cost of a NN with L2 regularization'''
    return cost + tf.losses.get_regularization_losses()
