#!/usr/bin/env python3
'''Module contains Neuron class'''
import numpy as np


class Neuron:
    '''class Neuron'''

    def __init__(self, nx):
        '''constructor'''
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
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
        '''
        Method calculates forward propagation
        of a neuron
        '''
        mul = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-mul))
        return self.__A

    def cost(self, Y, A):
        '''
        Method calculates cost of the model
        using logistic regression
        '''
        model = Y.shape[1]
        cost = -(1 / model) * np.sum(
                np.multiply(Y, np.log(A)) +
                np.multiply(1 - Y, np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        '''Method evaluates the neuron's predictions'''
        self.forward_prop(X)
        A = self.__A
        cost = self.cost(Y, A)
        return np.where(A >= 0.5, 1, 0), cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        '''Method calculates one pass of gradient descent'''
        model = Y.shape[1]
        dz = A - Y
        dw = (1 / model) * np.matmul(X, dz.T)
        db = (1 / model) * np.sum(dz)
        self.__W = self.__W - (alpha * dw.T)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        '''Method trains the neuron'''
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)
