#!/usr/bin/env python3
'''Module contains create_placeholders function'''
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
	'''Function returns two placeholders for the neural network'''
	x = tf.placeholder(name="x", shape=nx, dtype=tf.float32)
	y = tf.placeholder(name="y", shape=nx, dtype=tf.float32)
	return x, y
