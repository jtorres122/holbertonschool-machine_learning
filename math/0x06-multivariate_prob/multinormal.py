#!/usr/bin/env python3
'''Module contains multinormal class'''
import numpy as np


class MultiNormal:
    '''Class represents a Multivariate Normal distribution'''

    def __init__(self, data):
        '''Class Constructor'''
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        if data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')
        self.mean, self.cov = self.mean_cov(data)

    @staticmethod
    def mean_cov(X):
        """that calculates the mean and covariance of a data set"""
        mean = np.mean(X, axis=1, keepdims=True)
        cov = np.matmul((X - mean), (X - mean).T) / (X.shape[1] - 1)

        return mean, cov

    def pdf(self, x):
        '''Method calculates the PDF at a data point'''
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape != (d, 1):
            raise ValueError('x must have the shape ({}, 1)'.format(d))
        m = self.mean
        cov = self.cov
        bottom = np.sqrt(((2 * np.pi) ** d) * (np.linalg.det(cov)))
        inv = np.linalg.inv(cov)
        exp = (-.5 * np.matmul(np.matmul((x - m).T, inv), (x - m)))
        result = (1 / bottom) * np.exp(exp[0][0])
        return result
