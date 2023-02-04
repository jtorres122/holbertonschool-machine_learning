#!/usr/bin/env python3
'''Function contains gmm(X, k) function'''
import sklearn.mixture


def gmm(X, k):
    '''Function calculates a GMM from a dataset'''
    gauss = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi = gauss.weights_
    m = gauss.means_
    S = gauss.covariances_
    clss = gauss.predict(X)
    bic = gauss.bic(X)
    return pi, m, S, clss, bic
