#!/usr/bin/env python3
'''Module contains initialize(X, k) function'''
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    '''Function initializes variables for a Gaussian Mixture Model'''
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) != int or k < 1:
        return None, None, None
    n, d = X.shape
    pi = np.ones(k) / k
    cen, classes = kmeans(X, k)
    cov_mat = np.tile(np.identity(d), (k, 1, 1))
    return pi, cen, cov_mat
