#!/usr/bin/env python3
'''Module contains backward function'''
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    '''
    Function calculates the most likely sequence of
    hidden states for a hidden markov model
    '''
    try:
        N, M = Emission.shape
        T = Observation.shape[0]
        B = np.zeros((N, T))
        B[:, T - 1] = np.ones(N)
        for j in range(T - 2, -1, -1):
            for i in range(N):
                aux = Emission[:, Observation[j + 1]] * Transition[i, :]
                B[i, j] = np.dot(B[:, j + 1], aux)
        P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
        return P, B
    except Exception as e:
        return None, None
