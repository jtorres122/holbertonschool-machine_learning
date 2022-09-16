#!/usr/bin/env python3
'''Module contains convolve function'''
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    '''Function performs a convolution on images using multiple kernels'''
    m, h, w, c = images.shape
    kh, kw, c, nc = kernels.shape
    sh, sw = stride

    if padding == 'valid':
        ph = pw = 0
    elif padding == 'same':
        ph = int((((h - 1) * sh + kh - h) / 2) + 1)
        pw = int((((w - 1) * sw + kw - w) / 2) + 1)
    else:
        ph, pw = padding

    oh = int(((h + 2 * ph - kh) / sh) + 1)
    ow = int(((w + 2 * pw - kw) / sw) + 1)

    conv_img = np.zeros((m, oh, ow, nc))

    padded_imgs = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw),
                         (0, 0)), mode='constant')

    for y in range(oh):
        for x in range(ow):
            img_slice = padded_imgs[:, (y*sh):(y*sh)+kh, (x*sw):(x*sw)+kw, :]
            for z in range(nc):
                conv_img[:, y, x, z] = np.tensordot(img_slice,
                                                    kernels[:, :, :, z],
                                                    axes=c)

    return conv_img
