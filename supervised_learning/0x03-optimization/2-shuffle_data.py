#!/usr/bin/env python3
'''Module contains shuffle_data function'''
import numpy as np


def shuffle_data(X, Y):
    '''Shuffles the data points in two matrices the same way'''

    assert len(X) == len(Y)

    permutation = list(np.random.permutation(len(X)))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    return shuffled_X, shuffled_Y
