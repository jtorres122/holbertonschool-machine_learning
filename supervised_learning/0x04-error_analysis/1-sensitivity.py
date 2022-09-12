#!/usr/bin/env python3
'''Module contains sensitivity function'''
import numpy as np


def sensitivity(confusion):
    '''
    Function calculates the sensitivity for each
    class in a confusion matrix
    '''

    '''
    The number of correct positive predictions (TP) is
    divided by the total number of positives (P).
    '''
    return np.diag(confusion) / np.sum(confusion, axis=1)
