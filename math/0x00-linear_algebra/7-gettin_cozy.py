#!/usr/bin/env python3
'''Module contains a method that concatenates two matrices'''


def cat_matrices2D(mat1, mat2, axis=0):
    '''Function concatenates two matrices'''

    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        mat3 = mat1 + mat2

    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        mat3 = []
        for i in range(len(mat1)):
            mat3.append(mat1[i] + mat2[i])

    return mat3
