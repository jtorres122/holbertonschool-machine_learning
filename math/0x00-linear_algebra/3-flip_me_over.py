#!/usr/bin/env python3
'''Module contains a method that returns the transpose of a matrix'''


def matrix_transpose(matrix):
    '''Function returns the transpose of a matrix'''

    return [[matrix[j][i] for j in range(len(matrix))]
            for i in range(len(matrix[0]))]
