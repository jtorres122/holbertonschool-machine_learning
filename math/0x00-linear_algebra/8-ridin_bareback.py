#!/usr/bin/env python3
'''Module contains a method that performs matrix multiplication'''


def mat_mul(mat1, mat2):
    '''Function performs matrix multiplication'''

    result = [[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]

    for i in range(len(mat1)):
        if len(mat1[i]) != len(mat2):
            return None
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
