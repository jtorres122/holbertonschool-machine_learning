#!/usr/bin/env python3
import numpy as np
import tensorflow.compat.v1 as tf


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    '''Calculates one pass of gradient descent on the neural network'''
    m = Y.shape[1]
    dZ = {}
    for i in range(L, 0, -1):
        if i == L:
            dZ[i] = cache['A' + str(i)] - Y
        else:
            dZ[i] = np.matmul(weights['W' + str(i + 1)].T, dZ[i + 1]) * (cache['A' + str(i)] > 0)
        dW = np.matmul(dZ[i], cache['A' + str(i - 1)].T) / m
        db = np.sum(dZ[i], axis=1, keepdims=True) / m
        dW += lambtha / m * weights['W' + str(i)]
        weights['W' + str(i)] = weights['W' + str(i)] - alpha * dW
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * db
    return weights
