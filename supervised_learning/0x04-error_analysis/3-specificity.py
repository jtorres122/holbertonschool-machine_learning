#!/usr/bin/env python3
'''Module contains specificity function'''
import numpy as np


def specificity(confusion):
    '''
    Function calculates the specificity for each
    class in a confusion matrix
    '''

    '''
    The number of correct negative predictions (TN)
    divided by the total number of negatives (N).
    '''
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    TN = np.sum(confusion) - (TP + FP + FN)

    N = TN + FP

    return TN / N

