#!/usr/bin/env python3
'''Module contains early_stopping function'''
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    '''Function determines if you should stop gradient descent early'''
    if opt_cost - cost > threshold:
        return False, 0
    else:
        count += 1
        if count != patience:
            return(False, count)
    return True, count
