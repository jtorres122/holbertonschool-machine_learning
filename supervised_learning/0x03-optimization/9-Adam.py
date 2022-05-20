#!/usr/bin/env python3
'''Module contains update_variables_Adam function'''


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    '''Function that updates the variables using the Adam optimization algo'''
    v_t = beta1 * v + (1 - beta1) * grad
    s_t = beta2 * s + (1 - beta2) * grad * grad
    v_corr = v_t / (1 - beta1 ** t)
    s_corr = s_t / (1 - beta2 ** t)
    var_update = var - alpha * v_corr / (s_corr ** 0.5 + epsilon)
    return var_update, v_t, s_t
