#!/usr/bin/env python3
'''Module contains convolve_grayscale_valid function'''
import numpy as np


def convolve_grayscale_valid(images, kernel):
    '''Function performs a valid convolution on grayscale images'''
    m, hm, wm = images.shape
    hk, wk = kernel.shape
    ch = hm - hk + 1
    cw = wm - wk + 1
    convoluted = np.zeros((m, ch, cw))
    for h in range(ch):
        for w in range(cw):
            square = images[:, h: h + hk, w: w + wk]
            insert = np.sum(square * kernel, axis=1).sum(axis=1)
            convoluted[:, h, w] = insert
    return convoluted
