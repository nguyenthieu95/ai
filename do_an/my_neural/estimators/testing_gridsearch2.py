#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 11:53:06 2018

@author: thieunv
"""

from sklearn.base import BaseEstimator
from sklearn import tree
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.scorer import make_scorer


class custom_classifier(BaseEstimator):

  # Kmeans clustering model
    __clusters = None
  # decision tree model
    __tree = None
  # x library.
    __X = None
  # y library.
    __y = None
  # columns selected from pandas dataframe.
    __columns = None

    def fit(self,X, y,**kwargs):
        self.fit_kmeans(self.__X, self.__y)
        self.fit_decisiontree(self.__X, self.__y)

    def predict(self, X):
        result_kmeans = self.__clusters.predict(X)
        result_tree = self.__tree.predict(X)
        result = result_tree
        return np.array(result)

    def fit_kmeans(self, X, y):
        clusters = KMeans(n_clusters=4, random_state=0).fit(X)

    # the error center should have the lowest number of labels.(implementation not shown here)
        self.__clusters = clusters

    def fit_decisiontree(self, X, y):
        temp_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
        temp_tree.fit(X, y)
        self.__tree = temp_tree

    def seg_tree_hit_func(ground_truth, predictions):
        total_hit = 0
        total_number = 0
        for i in xrange(len(predictions)):
            if predictions[i] == 2:
                continue
            else:
                total_hit += 1 - abs(ground_truth[i] - predictions[i])
                total_number += 1.0
            print 'skipped: ', len(predictions) - total_number, '/', \
                len(predictions), 'instances'
        return (total_hit / total_number if total_number != 0 else 0)
    
    
    #make our own scorer
score = make_scorer(seg_tree_hit_func, greater_is_better=True)
scores = cross_val_score(custom_classifier(), X, Y, cv=7, scoring=score)

    
    
    
    
    
    