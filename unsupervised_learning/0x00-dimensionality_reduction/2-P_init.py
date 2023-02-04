#!/usr/bin/env python3
'''Module contains P_init(X, perplexity) function'''
import numpy as np


def P_init(X, perplexity):
    '''
    Function initializes all variables required
    to calculate the P affinities in t-SNE
    '''
    n = X.shape[0]
    X1 = X[:, :, None]
    D = ((X1 - X1.T) ** 2).sum(1)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    H = np.log2(perplexity)
    return D, P, beta, H
