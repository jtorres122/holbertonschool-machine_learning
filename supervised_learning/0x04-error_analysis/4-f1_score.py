#!/usr/bin/env python3
'''Module contains specificity function'''
import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    '''Function calculates the F1 score of a confusion matrix'''
    prec = precision(confusion)
    sens = sensitivity(confusion)
    return 2 * (prec * sens) / (prec + sens)
