#!/usr/bin/env python3
'''Module contains variance(X, C) function'''
import numpy as np


def variance(X, C):
    '''Function calculates the total intra-cluster variance for a data set'''
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    try:
        distances = np.sum(np.square(X - C[:, np.newaxis]), axis=2)
        min_distances = np.min(distances, axis=0)
        intra = np.sum(min_distances)
        return intra
    except Exception as e:
        return None
