#!/usr/bin/env python3
'''Module contains create_confusion_matrix function'''
import numpy as np


def create_confusion_matrix(labels, logits):
    '''Function creates a confusion matrix'''

    return np.matmul(labels.T, logits)
