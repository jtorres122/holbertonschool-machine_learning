#!/usr/bin/env python3
'''Module contains the function norm_constants'''
import numpy as np


def normalization_constants(X):
    '''
    Function calculates the mean and standard deviation
    of each feature, respectively.
    '''
    mean = np.mean(X, axis=0)
    standard_dev = np.std(X, axis=0)
    return mean, standard_dev
