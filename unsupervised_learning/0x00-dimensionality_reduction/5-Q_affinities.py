#!/usr/bin/env python3
'''Module contains Q_affinities(Y) function'''
import numpy as np


def Q_affinities(Y):
    '''Function calculates the symmetric P affinities of a data set'''
    sum = np.sum(np.square(Y), 1)
    D = np.add(np.add(-2 * np.matmul(Y, Y.T), sum).T, sum)
    top = (1 + D) ** (-1)
    np.fill_diagonal(top, 0)
    bottom = np.sum(top)
    Q = top / bottom
    return Q, top
