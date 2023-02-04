#!/usr/bin/env python3
'''Module contains pdf(X, m, S) function'''
import numpy as np


def pdf(X, m, S):
    '''Function calculates the probability density function'''
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None
    n, d = X.shape
    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    front = 1 / np.sqrt((2 * np.pi) ** d * det)
    part1 = np.matmul((-(X - m) / 2), inv)
    part2 = np.matmul(part1, (X - m).T).diagonal()
    expo = np.exp(part2)
    pdf = expo * front
    P = np.where(pdf < 1e-300, 1e-300, pdf)
    return P
