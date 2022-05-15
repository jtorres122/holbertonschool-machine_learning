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
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError('layers must be a list of positive integers')
            self.weights['W' + str(i + 1)] =\
                np.random.normal(size=(layers[i], nx)) * np.sqrt(2 / nx)
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
            nx = layers[i]
