#!/usr/bin/env python3
'''Module contains the batch_norm function'''
import numpy as np


def batch_norm(Z, beta, gamma, epsilon):
    '''
    Function normalizes an unactivated output
    of a neural network using batch normalization
    '''
    batch_mean = np.mean(Z, axis=0)
    batch_var = np.var(Z, axis=0)
    return ((Z - batch_mean) / np.sqrt(batch_var + epsilon)) * gamma + beta
