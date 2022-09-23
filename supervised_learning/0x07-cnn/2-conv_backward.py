#!/usr/bin/env python3
'''Module contains conv_backward function'''
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    '''
    Function performs back propagation over a
    convolutional layer of a neural network
    '''
    k_h, k_w, c_prev, c_new = W.shape
    _, h_new, w_new, c_new = dZ.shape
    m, h_x, w_x, c_prev = A_prev.shape
    s_h, s_w = stride
    x = A_prev

    if padding == 'valid':
        p_h = 0
        p_w = 0

    if padding == 'same':
        p_h = np.ceil(((s_h*h_x) - s_h + k_h - h_x) / 2)
        p_h = int(p_h)
        p_w = np.ceil(((s_w*w_x) - s_w + k_w - w_x) / 2)
        p_w = int(p_w)

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    x_padded = np.pad(x, [(0, 0), (p_h, p_h), (p_w, p_w), (0, 0)],
                      mode='constant', constant_values=0)

    dW = np.zeros_like(W)
    dx = np.zeros(x_padded.shape)
    m_i = np.arange(m)
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c_new):
                    dx[i,
                        h*(stride[0]):(h*(stride[0]))+k_h,
                        w*(stride[1]):(w*(stride[1]))+k_w,
                        :] += dZ[i, h, w, f] * W[:, :, :, f]
                    dW[:, :,
                        :, f] += x_padded[i,
                                          h*(stride[0]):(h*(stride[0]))+k_h,
                                          w*(stride[1]):(w*(stride[1]))+k_w,
                                          :] * dZ[i, h, w, f]
    if padding == 'same':
        dx = dx[:, p_h:-p_h, p_w:-p_w, :]
    else:
        dx = dx

    return dx, dW, db
