#!/usr/bin/env python3
'''Module contains the function norm_constants'''


def normalization_constants(X):
    '''
    Function calculates the mean and standard deviation
    of each feature, respectively.
    '''
    mean = X.mean(axis=0)
    standard_dev = X.std(axis=0)
    return mean, standard_dev
