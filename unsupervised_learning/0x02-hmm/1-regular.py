#!/usr/bin/env python3
'''Module contains regular(P) function'''
import numpy as np


def regular(P):
    '''
    Function determines the steady state probabilities
    of a regular markov chain
    '''
    if len(P.shape) != 2:
        return None
    n, _ = P.shape
    if n != P.shape[1]:
        return None
    evals, evecs = np.linalg.eig(P.T)
    state = evecs / evecs.sum()
    times = np.where(np.isclose(evals, 1))[0]
    if len(times) == 1:
        num = times[0]
        result = state[:, [num]]
        return result.reshape(1, n)
    return None
