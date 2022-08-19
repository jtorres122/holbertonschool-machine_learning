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
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        '''Calculates the cost of the model using logistic regression'''
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y)
                               * (np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        '''Function evaluates the neuron's predictions'''
        A = self.forward_prop(X)
        a = np.where(A <= 0.5, 0, 1)
        return A, self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        '''Calculates one pass of gradient descent on the neuron'''
        m = Y.shape[1]
        dw = np.matmul(A - Y, X.T) / m
        db = np.sum(A - Y) / m

        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)
