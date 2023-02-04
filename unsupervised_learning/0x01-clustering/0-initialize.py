#!/usr/bin/env python3
'''Module contains initialize(X, k) function'''
import numpy as np


def initialize(X, k):
    '''Function initializes cluster centroids for K-means'''
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k < 1:
        return None
    n, d = X.shape
    small = np.min(X, axis=0)
    large = np.max(X, axis=0)
    new = np.random.uniform(
        low=small,
        high=large,
        size=(k, d)
    )
    return new
