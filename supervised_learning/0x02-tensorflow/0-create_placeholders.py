#!/usr/bin/env python3
'''Module contains create_placeholders function'''
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    '''Function returns two placeholders, x and y'''
    x = tf.placeholder(tf.float32, [None, nx], name='x')
    y = tf.placeholder(tf.float32, [None, classes], name='y')
    return x, y
