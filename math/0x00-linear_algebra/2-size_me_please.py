#!/usr/bin/env python3
'''Module contains matrix_shape function'''


def matrix_shape(matrix):
    '''Function calculates the shape of a matrix'''
    shape = []
    while type(matrix) == list:
        shape.append(len(matrix))
        matrix = matrix[0]

    return shape
