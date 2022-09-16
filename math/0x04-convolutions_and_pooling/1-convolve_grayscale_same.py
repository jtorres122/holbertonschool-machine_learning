#!/usr/bin/env python3
'''Module contains convolve_grayscale_same function'''
import numpy as np


def convolve_grayscale_same(images, kernel):
    '''Function performs a same convolution on grayscale images'''
    kh, kw = kernel.shape

    ph = int(np.ceil((kh - 1) / 2))
    pw = int(np.ceil((kw - 1) / 2))

    conv_img = np.zeros(images.shape)

    padded_imgs = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                         mode='constant')

    for y in range(images.shape[1]):
        for x in range(images.shape[2]):
            img_slice = padded_imgs[:, y:(y + kh), x:(x + kw)]
            conv_img[:, y, x] = np.tensordot(img_slice, kernel)

    return conv_img
