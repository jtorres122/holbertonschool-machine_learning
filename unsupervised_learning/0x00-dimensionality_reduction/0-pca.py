#!/usr/bin/env python3
'''Module contains pca(X, var=0.95) function'''
import numpy as np


def pca(X, var=0.95):
    '''Function performs PCA on a data'''
    K = 1
    U, S, V = np.linalg.svd(X, full_matrices=True)
    for x in range(X.shape[1]):
        varian = (np.sum(S[:K]) / np.sum(S))
        if varian >= var:
            return V[:K].T
        K += 1
