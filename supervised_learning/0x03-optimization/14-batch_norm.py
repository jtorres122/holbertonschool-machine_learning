#!/usr/bin/env python3
'''Module contains create_batch_norm_layer function'''
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras


def create_batch_norm_layer(prev, n, activation):
    '''Function'''
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = keras.layers.Dense(units=n, kernel_initializer=w)
    mean, variance = tf.nn.moments(layer(prev), axes=[0])
    gamma = tf.Variable(tf.ones((1, n)), trainable=True)
    beta = tf.Variable(tf.zeros((1, n)), trainable=True)
    r = tf.nn.batch_normalization(layer(prev), mean, variance,
                                  beta, gamma, 1e-8)
    return activation(r)
