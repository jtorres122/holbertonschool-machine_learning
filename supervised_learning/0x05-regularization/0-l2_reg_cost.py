#!/usr/bin/env python3
'''Module contains l2_reg_cost function'''
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    '''Function calculates the cost of a NN with L2 regularization'''

    norm = 0
    for w in weights.keys():
        norm += np.linalg.norm(weights[w]) ** 2

    return cost + ((lambtha / (2 * m)) * norm)
