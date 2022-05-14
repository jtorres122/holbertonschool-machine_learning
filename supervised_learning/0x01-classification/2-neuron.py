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
