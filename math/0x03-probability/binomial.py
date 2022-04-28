#!/usr/bin/env python3
'''Module contains Binomial class'''


class Binomial:
    '''
    Class represents a Binomial distribution
    '''

    def __init__(self, data=None, n=1, p=0.5):
        '''Class instantiator'''
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p < 0 and p > 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = int(n)
            self.p = float(p)
        else:
            if isinstance(data, list) is False:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            variance = sum((i - mean) ** 2 for i in data) / len(data)
            self.p = - (variance / mean) + 1
            self.n = round(mean / self.p)
            self.p = mean / self.n
