#!/usr/bin/env python3
'''Module contains poisson class'''


class Poisson:
    '''
    Class represents a poisson distribution
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
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        '''
        Method calculates the value of the PMF
        for a given number of “successes”
        '''
        if isinstance(k, int) is False:
            self.k = int(k)
        if k < 0:
            return 0
        return

    def cdf(self, k):
        '''
        Method calculates the value of the CDF
        for a given number of “successes”
        '''
        if isinstance(k, int) is False:
            self.k = int(k)
        if k < 0:
            return 0
        return
