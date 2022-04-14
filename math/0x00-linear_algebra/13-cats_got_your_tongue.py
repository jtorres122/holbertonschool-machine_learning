#!/usr/bin/env python3
'''
Module contains a method that concatenates
two matrices along a specific axis
'''
import numpy as np


def np_cat(mat1, mat2, axis=0):
    '''Function'''

    return np.concatenate((mat1, mat2), axis)
