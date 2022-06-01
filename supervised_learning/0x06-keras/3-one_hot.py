#!/usr/bin/env python3
'''Module contains one_hot function'''
import tensorflow.keras as K


def one_hot(labels, classes=None):
    '''Function converts vector labels to one-hot matrices'''
    return K.utils.to_categorical(labels, num_classes=classes)
