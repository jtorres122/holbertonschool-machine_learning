#!/usr/bin/env python3
'''Module contains matrix_transpose function'''


def matrix_transpose(matrix):
    '''Function returns the transpose of a matrix'''
    result = [[matrix[j][i] for j in range(len(matrix))]
              for i in range(len(matrix[0]))]
    return result
