#!/usr/bin/env python3
'''Module contains create_placeholders function'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_placeholders(nx, classes):
    '''Function returns two placeholders for the neural network'''
    x = tf.placeholder(name="x", shape=(None, nx), dtype=tf.float32)
    y = tf.placeholder(name="y", shape=(None, classes), dtype=tf.float32)
    return x, y
