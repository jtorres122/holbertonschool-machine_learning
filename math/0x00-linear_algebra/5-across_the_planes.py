#!/usr/bin/env python3
'''Module contains add_matrices2D function'''


def add_matrices2D(mat1, mat2):
    '''Function adds two matrices element wise'''
    mat_sum = []
    if len(mat1[0]) != len(mat2[0]):
        return None

    for row in range(len(mat1)):
        newList = []
        for col in range(len(mat1[row])):
            newList.append((mat1[row][col] + mat2[row][col]))
        mat_sum.append(newList)

    return mat_sum
