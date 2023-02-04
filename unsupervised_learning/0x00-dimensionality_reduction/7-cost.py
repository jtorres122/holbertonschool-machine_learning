#!/usr/bin/env python3
'''Module contains cost(P, Q) function'''
import numpy as np


def cost(P, Q):
    '''Function calculates the cost of the t-SNE transformation'''
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)
    C = np.sum(P * np.log(P / Q))
    return C
