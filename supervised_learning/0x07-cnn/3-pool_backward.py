#!/usr/bin/env python3
'''Module contains pool_backward function'''
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    '''
    Function performs back propagation over a
    pooling layer of a neural network
    '''
    k_h, k_w = kernel_shape
    m, h_new, w_new, c_new = dA.shape
    m, h_x, w_x, c_prev = A_prev.shape
    s_h, s_w = stride

    dx = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c_new):
                    if mode == 'max':
                        tmp = A_prev[i, h*s_h:k_h+(h*s_h),
                                     w*s_w:k_w+(w*s_w), f]
                        mask = (tmp == np.max(tmp))
                        dx[i,
                            h*(s_h):(h*(s_h))+k_h,
                            w*(s_w):(w*(s_w))+k_w,
                            f] += dA[i, h, w, f] * mask
                    if mode == 'avg':
                        dx[i,
                            h*(s_h):(h*(s_h))+k_h,
                            w*(s_w):(w*(s_w))+k_w,
                            f] += (dA[i, h, w, f])/k_h/k_w

    return dx
