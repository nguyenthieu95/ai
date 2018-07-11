#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 10:13:13 2018
@author: thieunv
"""

import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def elu(x, alpha=1):
    return np.where(x < 0, alpha * (np.exp(x) - 1), x)

def relu(x):
    return np.max(x, 0)

def tanh(x):
    return np.tanh(x)



def legendre(data):
    x1 = data
    x2 = 3 / 2 * np.power(data, 2) - 1 / 2
    x3 = 1 / 3 * (5 * data * x2 - 2 * x1)
    x4 = 1 / 4 * (7 * data * x3 - 3 * x2)
    x5 = 1 / 5 * (9 * data * x4 - 4 * x3)
    x6 = 1 / 6 * (11 * data * x5 - 5 * x4)

    return [x2, x3, x4, x5, x6]

def laguerre(data):
    x1 = -data + 1
    x2 = np.power(data, 2) / 2 - 2 * data + 1
    x3 = 1 / 3 * ((5 - data) * x2 - 2 * x1)
    x4 = 1 / 4 * ((7 - data) * x3 - 3 * x2)
    x5 = 1 / 5 * ((9 - data) * x4 - 4 * x3)
    x6 = 1 / 6 * ((11 - data) * x5 - 5 * x4)

    return [x2, x3, x4, x5, x6]

def chebyshev(data):
    x2 = 2 * np.power(data, 2) - 1
    x3 = 4 * np.power(data, 3) - 3 * data
    x4 = 8 * np.power(data, 4) - 8 * np.power(data, 2) + 1
    x5 = 2 * data * x4 - x3
    x6 = 2 * data * x5 - x4

    return [x2, x3, x4, x5, x6]

def powerseries(data):
    x2 = np.power(data, 2)
    x3 = np.power(data, 3)
    x4 = np.power(data, 4)
    x5 = np.power(data, 5)
    x6 = np.power(data, 6)

    return [x2, x3, x4, x5, x6]
