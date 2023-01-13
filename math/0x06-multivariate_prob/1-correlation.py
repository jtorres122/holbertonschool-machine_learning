#!/usr/bin/env python3
'''Module contains correlation(C) function'''
import numpy as np


def correlation(C):
    '''Function calculates a correlation matrix'''

    if not isinstance(C, np.ndarray):
        raise TypeError('C must be a numpy.ndarray')
    if C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')

    return np.corrcoef(C)