#!/usr/bin/env python3
'''Module contains l2_reg_gradient_descent function'''
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    '''
    Function updates the weights and biases of a NN
    using gradient descent with L2 regularization
    '''

    m = Y.shape[1]
    dz = cache["A" + str(L)] - Y
    for idx in range(L, 0, -1):
        A_prev = cache["A" + str(idx - 1)]
        dw = np.matmul(dz, A_prev.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        inv = 1 - np.square(A_prev)
        dz = np.matmul(weights["W" + str(idx)].T, dz) * inv
        reg = (1 - lambtha * alpha / m)
        weights["W" + str(idx)] = reg * weights["W" + str(idx)] - alpha * dw
        weights["b" + str(idx)] -= alpha * db
