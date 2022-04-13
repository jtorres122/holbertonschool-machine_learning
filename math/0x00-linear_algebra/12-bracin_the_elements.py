#!/usr/bin/env python3
'''
Module contains a method that performs
element-wise addition, subtraction, multiplication, and division
'''


def np_elementwise(mat1, mat2):
    '''
    Function performs element-wise addition, subtraction,
    multiplication, and division
    '''

    return (mat1 + mat2,
            mat1 - mat2,
            mat1 * mat2,
            mat1 / mat2)
