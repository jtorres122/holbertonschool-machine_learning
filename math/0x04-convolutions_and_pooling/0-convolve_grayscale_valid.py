#!/usr/bin/env python
'''Module contains convolve_grayscale_valid function'''
import numpy as np


def convolve_grayscale_valid(images, kernel):
    '''Function performs a valid convolution on grayscale images'''
    m, h, w = images.shape
    kh, kw = kernel.shape

    conv_w = (w - kw) + 1
    conv_h = (h - kh) + 1

    conv_img = np.zeros((m, conv_w, conv_h))

    for y in range(conv_h):
        for x in range(conv_w):
            img_slice = images[:, y:y + kh, x:x + kw]
            conv_img[:, y, x] = np.tensordot(img_slice, kernel)

    return conv_img
