#!/usr/bin/env python3
'''Module contains convolve_grayscale function'''
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    '''Function performs a convolution on grayscale images'''
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'valid':
        ph = pw = 0
    elif padding == 'same':
        ph = int((((h - 1) * sh + kh - h) / 2) + (kh % 2 == 0))
        pw = int((((w - 1) * sw + kw - w) / 2) + (kw % 2 == 0))
    else:
        ph, pw = padding

    oh = int(((h + 2 * ph - kh) / sh) + 1)
    ow = int(((w + 2 * pw - kw) / sw) + 1)

    conv_img = np.zeros((m, oh, ow))

    padded_imgs = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                         mode='constant')

    for y in range(oh):
        for x in range(ow):
            img_slice = padded_imgs[:, y * sh:(y * sh) + kh,
                                    (x * sw):(x * sw) + kw]
            conv_img[:, y, x] = np.tensordot(img_slice, kernel)

    return conv_img