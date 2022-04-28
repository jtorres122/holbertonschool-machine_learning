#!/usr/bin/env python3
'''Module contains exponential class'''


class Exponential:
    '''
    Class represents a exponential distribution
    '''

    def __init__(self, data=None, lambtha=1.):
        '''Class constructor'''
        if data is None:
            if lambtha > 0:
                self.lambtha = float(lambtha)
            else:
                raise ValueError('lambtha must be a positive value')
        else:
            if isinstance(data, list) is False:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = len(data) / sum(data)

    def pdf(self, x):
        '''
        Method calculates the value of the PDF
        for a given time period
        '''
        if x < 0:
            return 0
        return

    def cdf(self, x):
        '''
        Method calculates the value of the CDF
        for a given time period
        '''
        if x < 0:
            return 0
        return
