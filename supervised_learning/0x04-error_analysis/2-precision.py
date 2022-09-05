#!/usr/bin/env python3
'''Module contains precision function'''
import numpy as np


def precision(confusion):
    '''
    Function calculates the precision for each
    class in a confusion matrix
    '''

    '''
    The number of correct positive predictions (TP)
    divided by the total number of positive predictions (TP + FP).
    '''
    return np.diag(confusion) / np.sum(confusion, axis=0)
