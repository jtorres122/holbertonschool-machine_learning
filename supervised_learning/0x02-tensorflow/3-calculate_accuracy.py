#!/usr/bin/env python3
'''Module contains calculate_accuracy function'''
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    '''Calculates the accuracy of a prediction'''
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy
