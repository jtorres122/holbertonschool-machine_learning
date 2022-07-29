#!/usr/bin/env python3
'''Module contains add_arrays function'''


def add_arrays(arr1, arr2):
    '''Function adds two arrays element wise'''
    arr_sum = []
    if len(arr1) != len(arr2):
        return None

    for idx in range(len(arr1)):
        arr_sum.append(arr1[idx] + arr2[idx])

    return arr_sum
