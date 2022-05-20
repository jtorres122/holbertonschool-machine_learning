#!/usr/bin/env python3
'''Module containing the Momentum optimization function'''


def update_variables_momentum(alpha, beta1, var, grad, v):
    '''
    Function updates a variable using the gradient descent
    with momentum optimization algorithm
    '''
    v = beta1 * v + (1 - beta1) * grad
    var -= alpha * v
    return var, v
