#!/usr/bin/env python3
'''Module contains DeepNeuralNetwork class'''
import numpy as np


class DeepNeuralNetwork:
    '''class DeepNeuralNetwork'''

    def __init__(self, nx, layers):
        '''class instantiator'''
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list:
            raise TypeError('layers must be a list of positive integers')
        if not layers:
            raise TypeError('layers must be a list of positive integers')
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError('layers must be a list of positive integers')
            self.__weights['W' + str(i + 1)] =\
                np.random.normal(size=(layers[i], nx)) * np.sqrt(2 / nx)
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
            nx = layers[i]

    @property
    def L(self):
        '''L getter'''
        return self.__L

    @property
    def cache(self):
        '''cache getter'''
        return self.__cache

    @property
    def weights(self):
        '''weights getter'''
        return self.__weights

    def forward_prop(self, X):
        '''Method calculates forward propagation'''
        self.__cache['A0'] = X
        for i in range(self.__L):
            W = self.__weights['W' + str(i + 1)]
            b = self.__weights['b' + str(i + 1)]
            A = self.__cache['A' + str(i)]
            Z = np.matmul(W, A) + b
            self.__cache['A' + str(i + 1)] = 1 / (1 + np.exp(-Z))
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        '''Method calculates cost of model using logistic regression'''
        model = Y.shape[1]
        cost = -(1 / model) * np.sum(
            np.multiply(Y, np.log(A)) +
            np.multiply(1 - Y, np.log(1.0000001 - A)))
        return cost
