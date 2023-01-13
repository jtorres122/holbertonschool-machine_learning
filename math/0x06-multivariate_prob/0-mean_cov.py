#!/usr/bin/env python3
'''Module contains def mean_cov(X) function'''
import numpy as np


def mean_cov(X):
    '''Function calculates the mean and covariance of a data set'''

    if len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    if X.shape[0] < 2:
        raise ValueError('X must contain multiple data points')

    means = []
    for set in X:
        buffer = []
        for idx in range(len(set)):
            buffer.append(set[idx])
        means.append(buffer)

    mean = np.array(np.mean(means, axis=0))

    cov = []
    for j in range(len(mean)):
        covs = []
        for k in range(len(mean)):
            terms = ((means[i][j] - mean[j])
                     * (means[i][k] - mean[k]) for i in range(len(means)))
            covariance = sum(terms) / len(means)
            covs.append(covariance)
        cov.append(covs)

    return mean, np.array(cov)
