#!/usr/bin/env python3
'''Module contains create_momentum_op function'''
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    '''
    Function creates the training operation for a neural network in tensorflow
    using the gradient descent with momentum optimization algorithm
    '''
    var = tf.trainable_variables()[0]
    grad = tf.gradients(loss, var)[0]
    v = tf.Variable(tf.zeros_like(var), trainable=False)
    v_op = update_variables_momentum(alpha, beta1, var, grad, v)
    return v_op


def update_variables_momentum(alpha, beta1, var, grad, v):
    '''
    Function updates a variable using the gradient descent
    with momentum optimization algorithm
    '''
    v = beta1 * v + (1 - beta1) * grad
    var -= alpha * v
    return var, v
