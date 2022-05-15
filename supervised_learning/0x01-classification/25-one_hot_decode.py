#!/usr/bin/env python3
'''Module contains one_hot_decode function'''
import numpy as np


def one_hot_decode(one_hot):
    '''
    Function converts a one-hot
    matrix into a vector of labels
    '''
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    if np.max(one_hot) != 1 or np.min(one_hot) != 0:
        return None
    return np.argmax(one_hot, axis=1)
