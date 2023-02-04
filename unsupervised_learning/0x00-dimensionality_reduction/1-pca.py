#!/usr/bin/env python3
'''Module contains pca(X, ndim) function'''
import numpy as np


def pca(X, ndim):
    '''Function performs PCA on a dataset'''
    n, d = X.shape
    X1 = X - np.mean(X, axis=0)
    U, S, V = np.linalg.svd(X1, full_matrices=True)
    return np.matmul(X1, V[:ndim].T)
