#!/usr/bin/env python3
'''Module contains pool function'''
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    '''Function performs pooling on images'''
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = int((h - kh) / sh) + 1
    ow = int((w - kw) / sw) + 1

    conv_img = np.zeros((m, oh, ow, c))

    for y in range(oh):
        for x in range(ow):
            img_slice = images[:, (y * sh):(y * sh) + kh,
                               (x * sw):(x * sw) + kw, :]
            if mode == 'max':
                conv_img[:, y, x, :] = np.max(img_slice, axis=(1, 2))
            else:
                conv_img[:, y, x, :] = np.average(img_slice, axis=(1, 2))

    return conv_img
