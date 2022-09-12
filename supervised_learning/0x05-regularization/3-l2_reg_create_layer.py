#!/usr/bin/env python3
'''Module contains l2_reg_create_layer function'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def l2_reg_create_layer(prev, n, activation, lambtha):
    '''Function creates a tensorflow layer that includes L2 regularization'''
    weight = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    reg = tf.keras.regularizers.L2(lambtha)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=weight,
                            kernel_regularizer=reg)
    return layer(prev)
