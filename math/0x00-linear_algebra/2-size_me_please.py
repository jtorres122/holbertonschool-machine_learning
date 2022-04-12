#!/usr/bin/env python3
'''Module contains a method that calculates the shape of a matrix'''


def matrix_shape(matrix):

    return [len(matrix)] + matrix_shape(matrix[0])\
            if(type(matrix) == list) else []
