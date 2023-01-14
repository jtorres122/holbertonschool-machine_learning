#!/usr/bin/env python3
'''Module contains def mean_cov(X) function'''
import numpy as np


def mean_cov(X):
    '''Function calculates the mean and covariance of a data set'''

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    if X.shape[0] < 2:
        raise ValueError('X must contain multiple data points')

    mean = np.mean(X, axis=0, keepdims=True)
    cov = np.matmul((X - mean).T, (X - mean)) / (X.shape[0] - 1)

    return mean, cov
