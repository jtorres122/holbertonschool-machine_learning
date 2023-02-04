#!/usr/bin/env python3
'''Module contains kmeans(X, k) function'''
import sklearn.cluster


def kmeans(X, k):
    '''Function performs K-means on a dataset'''
    kmean = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C_means = kmean.cluster_centers_
    classes = kmean.labels_
    return C_means, classes
