#!/usr/bin/env python3
'''Module contains mat_mul function'''


def mat_mul(mat1, mat2):
    '''Function performs matrix multiplication'''
    if (len(mat1[0]) == len(mat2)):
        return [[sum(a*b for a, b in zip(mat1_row, mat2_col))
                for mat2_col in zip(*mat2)] for mat1_row in mat1]
    else:
        return None
