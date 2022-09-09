#!/usr/bin/env python3
'''Module contains dropout_create_layer function'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def dropout_create_layer(prev, n, activation, keep_prob):
    '''Function creates a layer of a neural network using dropout'''
    weight = tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))
    dropout = tf.layers.Dropout(1 - keep_prob)
    layer = tf.layers.Dense(units=n, activation=activation,
                        kernel_initializer=weight, kernel_regularizer=dropout)
    return layer(prev)
