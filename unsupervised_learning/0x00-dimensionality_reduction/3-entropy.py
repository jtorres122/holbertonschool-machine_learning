#!/usr/bin/env python3
'''Module contains HP(Di, beta) function'''
import numpy as np


def HP(Di, beta):
    '''
    Function calculates the Shannon entropy and P
    affinities relative to a data point
    '''
    top = np.exp(-Di * beta)
    bottom = np.sum(top)
    pi = top / bottom
    hi = -np.sum(pi * np.log2(pi))
    return hi, pi
