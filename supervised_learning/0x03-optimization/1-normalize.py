#!/usr/bin/env python3
'''Module contains the function normalize'''


def normalize(X, m, s):
    '''
    Function normalizes the X dataset
    using the mean and standard deviation
    '''
    return (X - m) / s
