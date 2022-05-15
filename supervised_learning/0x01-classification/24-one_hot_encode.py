#!/usr/bin/env python3
'''Module contains one_hot_encode function'''
import numpy as np


def one_hot_encode(Y, classes):
    '''
    Function converts a numeric label
    vector into a one-hot matrix
    '''
    if type(Y) is not np.ndarray or len(Y.shape) != 1:
        return None
    if type(classes) is not int or classes <= 1:
        return None
    if np.max(Y) >= classes:
        return None
    if np.min(Y) < 0:
        return None
    n = Y.shape[0]
    one_hot = np.zeros((n, classes))
    for i in range(n):
        one_hot[i, Y[i]] = 1
    return one_hot
