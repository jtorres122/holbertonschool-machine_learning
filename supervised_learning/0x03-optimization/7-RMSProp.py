#!/usr/bin/env python3
'''Module contains update_variables_RMSProp function'''
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    '''Function updates a variable using the RMSProp optimization algorithm'''
    new_s = beta2 * s + (1 - beta2) * (grad ** 2)
    new_var = var - alpha * (grad / (epsilon + np.sqrt(new_s)))
    return new_var, new_s
