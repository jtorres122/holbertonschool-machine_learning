#!/usr/bin/env python3
'''Module contains NeuralNetwork class'''
import numpy as np


class NeuralNetwork:
    '''NeuralNetwork Class'''

    def __init__(self, nx, nodes):
        '''class instantiator'''
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        '''W1 getter'''
        return self.__W1

    @property
    def b1(self):
        '''b1 getter'''
        return self.__b1

    @property
    def A1(self):
        '''A1 getter'''
        return self.__A1

    @property
    def W2(self):
        '''W2 getter'''
        return self.__W2

    @property
    def b2(self):
        '''b2 getter'''
        return self.__b2

    @property
    def A2(self):
        '''A2 getter'''
        return self.__A2

    def forward_prop(self, X):
        '''
        Method calculates the forward propagation
        of the neural network
        '''
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        '''Method calculates cost of the model'''
        model = Y.shape[1]
        cost = -(1 / model) * np.sum(
                np.multiply(Y, np.log(A)) +
                np.multiply((1 - Y), np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        '''Method evaluates the neural network'''
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        return np.where(self.__A2 >= 0.5, 1, 0), cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        '''Method calculates one pass of gradient descent'''
        model = X.shape[1]
        dz2 = A2 - Y
        dw2 = (1 / model) * np.matmul(dz2, A1.T)
        db2 = (1 / model) * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = (1 / model) * np.matmul(dz1, X.T)
        db1 = (1 / model) * np.sum(dz1, axis=1, keepdims=True)
        self.__W1 -= (alpha * dw1)
        self.__b1 -= (alpha * db1)
        self.__W2 -= (alpha * dw2)
        self.__b2 -= (alpha * db2)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        '''Method trains the neural network'''
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
        return self.evaluate(X, Y)
