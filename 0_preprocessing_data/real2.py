#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 12:03:34 2018

@author: thieunv
"""

fullpath = "/home/thieunv/university/LabThayMinh/code/data/preprocessing/"
filename = "time_series_ram.csv"

import pandas as pd
from pandas import Series
from matplotlib import pyplot
from numpy import log
series = Series.from_csv(fullpath + filename, header=0)


#series.plot()
#pyplot.show()


series.hist()
pyplot.show()


#X = series.values
#split = len(X) / 4
#X1, X2, X3, X4 = X[0:split], X[split:2*split], X[2*split:3*split], X[3*split:]
#mean1, mean2, mean3, mean4 = X1.mean(), X2.mean(), X3.mean(), X4.mean()
#var1, var2, var3, var4 = X1.var(), X2.var(), X3.var(), X4.var()
#print('mean1=%f, mean2=%f, mean3=%f, mean4=%f' % (mean1, mean2, mean3, mean4))
#print('variance1=%f, variance2=%f, variance3=%f, variance4=%f' % (var1, var2, var3, var4))



#### To show hist of log(X) we need to remove all value equal 0.0
#X = series.values
#dataset = []
#for t in X:
#    if t != 0.0:
#        dataset.append(t)
#X = pd.Series(dataset)
#X = log(X)
#pyplot.hist(X)
#pyplot.show()



### The stronger method to check your time series data is stational or non-stational
#The Augmented Dickey-Fuller test is a type of statistical test called a unit root test.
#it determines how strongly a time series is defined by a trend.
# It uses an autoregressive model and optimizes an information criterion across multiple different lag values.
# Kiem dinh gia thuyet. 
# Gia su time series bieu dien bang unit root (not stationary - gia tri phu thuoc thoi gian)
# Gia thuyet H0: Neu excepted -> Non-stationary
# Gia thuyet thay the H1: Stationary
# Gia tri p-value > 0.05% (Chap nhan H0 -> Non-stationary)
# p-value <= 0.05% (Tu choi H0 -> Stationary)


##### The stronger method to check time series using Kiem dinh gia thuyet
#from pandas import Series
#from statsmodels.tsa.stattools import adfuller
#series = Series.from_csv(fullpath + filename, header=0)
#X = series.values
#
#dataset = []
#for t in X:
#    if t != 0.0:
#        dataset.append(t)
#X = pd.Series(dataset)
#
#result = adfuller(X)        # Ham nay lam tat ca viec kiem dinh
#print('ADF Statistic: %f' % result[0])
#print('p-value: %f' % result[1])
#print('Critical Values:')
#for key, value in result[4].items():
#	print('\t%s: %.3f' % (key, value))




