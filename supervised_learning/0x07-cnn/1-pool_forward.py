#!/usr/bin/env python3
'''Module contains pool_forward function'''
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    '''
    Function performs forward propagation over
    a pooling layer of a neural network
    '''
    m, h_prev, w_prev, c_prev = A_prev.shape
    k_h, k_w = kernel_shape

    out_h = int(((h_prev - k_h) / (stride[0])) + 1)
    out_w = int(((w_prev - k_w) / (stride[1])) + 1)
    output_conv = np.zeros((m, out_h, out_w, c_prev))
    m_A_prev = np.arange(0, m)

    for i in range(out_h):
        for j in range(out_w):
            if mode == 'max':
                output_conv[m_A_prev, i, j] = np.max(
                    A_prev[
                        m_A_prev,
                        i*(stride[0]):k_h+(i*(stride[0])),
                        j*(stride[1]):k_w+(j*(stride[1]))], axis=(1, 2))
            if mode == 'avg':
                output_conv[m_A_prev, i, j] = np.mean(
                    A_prev[
                        m_A_prev,
                        i*(stride[0]):k_h+(i*(stride[0])),
                        j*(stride[1]):k_w+(j*(stride[1]))], axis=(1, 2))

    return output_conv
