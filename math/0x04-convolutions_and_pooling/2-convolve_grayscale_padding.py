#!/usr/bin/env python3
'''Module contains convolve_grayscale_padding function'''
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    '''
    Function performs a convolution on grayscale
    images with custom padding
    '''
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding[0], padding[1]

    oh = h + 2 * ph - kh + 1
    ow = w + 2 * pw - kw + 1
    conv_img = np.zeros((m, oh, ow))

    padded_imgs = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                         mode='constant')

    for y in range(oh):
        for x in range(ow):
            img_slice = padded_imgs[:, y:(y + kh), x:(x + kw)]
            conv_img[:, y, x] = np.tensordot(img_slice, kernel)

    return conv_img
