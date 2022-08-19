#!/usr/bin/env python3
'''Module contains class Neuron'''
import numpy as np


class Neuron:
    '''Class defines a single neuron performing binary classification'''

    def __init__(self, nx):
        '''Class constructor'''
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''W getter'''
        return self.__W

    @property
    def b(self):
        '''b getter'''
        return self.__b

    @property
    def A(self):
        '''A getter'''
        return self.__A

    def forward_prop(self, X):
        '''Function calculates the forward propagation of the neuron'''
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        '''Calculates the cost of the model using logistic regression'''
        m = Y.shape[1]
        cost = -1 / m * np.sum(np.dot(np.log(A), Y.T) +
                               np.dot(np.log(1.0000001 - A), (1 - Y.T)))
        return cost
