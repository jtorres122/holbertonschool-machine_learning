#!/usr/bin/env python3
'''Module contains DeepNeuralNetwork class'''
import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        '''Method evaluates the neural network's predictions'''
        A, __ = self.forward_prop(X)
        # The underscore is a throwaway variable
        cost = self.cost(Y, A)
        return np.round(A).astype(int), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        '''Method calculates one pass of gradient descent'''
        __, m = Y.shape
        dz = cache["A" + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            dw = (1 / m) * np.matmul(dz, self.__cache["A" + str(i - 1)].T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            self.__weights["W" + str(i)] -= (dw * alpha)
            self.__weights["b" + str(i)] -= (db * alpha)
            dz = np.matmul(self.__weights["W" + str(i)].T, dz) *\
                (cache["A" + str(i - 1)] * (1 - cache["A" + str(i - 1)]))
        return self.__weights

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        '''
        Method trains the deep NN by updating
        the private attributes __weights and __cache
        '''
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if step < 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        cost_list = []
        for i in range(iterations + 1):
            __, cost = self.evaluate(X, Y)
            self.gradient_descent(Y, self.cache, alpha)
            if i % step == 0 or i == iterations:
                cost_list.append(cost)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
        if graph is True:
            plt.plot(np.arange(0, iterations + 1, step), cost_list)
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.show()
        return self.evaluate(X, Y)
