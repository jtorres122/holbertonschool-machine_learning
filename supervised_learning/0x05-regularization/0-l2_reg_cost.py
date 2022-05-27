#!/usr/bin/env python3
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    '''Calculates the cost of a neural network with L2 regularization'''
    for i in range(1, L + 1):
        cost += lambtha / (2 * m) * np.sum(weights['W' + str(i)] ** 2)
    return cost
