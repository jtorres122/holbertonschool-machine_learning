#!/usr/bin/env python3
'''Module contains cat_matrices2D function'''


def cat_matrices2D(mat1, mat2, axis=0):
    '''Function concatenates two matrices along a specific axis'''
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return mat1 + mat2
    if axis == 1 and len(mat1) == len(mat2):
        cattedmat = []
        for row1, row2 in zip(mat1, mat2):
            cattedmat.append(row1 + row2)
        return cattedmat
    return None
