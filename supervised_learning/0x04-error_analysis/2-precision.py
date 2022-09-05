#!/usr/bin/env python3
'''Module contains precision function'''
import numpy as np


def precision(confusion):
    '''
    Function calculates the precision for each
    class in a confusion matrix
    '''
    return np.diag(confusion) / np.sum(confusion, axis=0)
