#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 10:13:13 2018

@author: thieunv
"""

import numpy as np
from random import randint, uniform
from copy import deepcopy
from scipy.spatial import distance


def distance_func(a, b):
    return distance.euclidean(a, b)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def elu(x, alpha=1):
    return np.where(x < 0, alpha * (np.exp(x) - 1), x)

def relu(x):
    return np.max(x, 0)

def tanh(x):
    return np.tanh(x)


def get_random_input_vector(data=None):
    return deepcopy(data[randint(0, len(data)-1)])

def get_batch_data_next(trainX, trainY, index=None, batch_size=None):
    real_index = index*batch_size
    if (len(trainX) % batch_size != 0 and index == (len(trainX)/batch_size +1) ):
        return (trainX[real_index:], trainY[real_index:])
    elif (real_index == len(trainX)):
        return ([], [])
    else:
        return (trainX[real_index: (real_index+batch_size)], trainY[real_index: (real_index+batch_size)])

def get_mutate_vector_weight(vectorA, vectorB, mutation_id = 1):
    temp = []
    if mutation_id == 1:    # Lay trung binh cong
        for i in range(len(vectorA)):
            temp.append( (vectorA[i] + vectorB[i]) / 2 )
    if mutation_id == 2:    # Lay uniform
        for i in range(len(vectorA)):
            temp.append(uniform(vectorA[i], vectorB[i]))
    return np.array(temp)

