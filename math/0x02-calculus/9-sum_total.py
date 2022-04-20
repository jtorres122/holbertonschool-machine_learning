#!/usr/bin/env python3
'''Module contains a function that calculates sigma notation'''


def summation_i_squared(n):
    '''Function calculates sum of series'''
    '''
    Note: recursion does not work because
    task asks for NoneType to be returned
    '''

    if isinstance(n, int) is False or n <= 0:
        return None

    return (n * (n + 1) * (2 * n + 1)) / 6
