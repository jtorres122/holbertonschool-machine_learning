#!/usr/bin/env python3
'''Module contains Normal class'''


class Normal:
    '''
    Class represents a normal distribution
    '''

    def __init__(self, data=None, mean=0., stddev=1.):
        '''Class instantiator'''
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if isinstance(data, list) is False:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = sum(data) / len(data)
            self.stddev = (sum((x - self.mean)**2 for x in data)
                           / len(data)) ** 0.5

    def z_score(self, x):
        '''Method calculates the z-score of a given x-value'''
