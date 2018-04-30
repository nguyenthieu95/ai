#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 11:55:22 2018

@author: thieunv

- Non-stational data

"""

fullpath = "/home/thieunv/university/LabThayMinh/code/data/preprocessing/"
filename = "international-airline-passengers.csv"


#### Show point
#from pandas import Series
#from matplotlib import pyplot
#series = Series.from_csv(fullpath + filename, header=0)
#series.plot()
#pyplot.show()


#### Show distribution (basiclly normal distribution)
#from pandas import Series
#from matplotlib import pyplot
#series = Series.from_csv(fullpath + filename, header=0)
#series.hist()
#pyplot.show()


#### Calculate Mean and Variance to see it's stational or non-stational
#from pandas import Series
#series = Series.from_csv(fullpath + filename, header=0)
#X = series.values
#split = len(X) / 2
#X1, X2 = X[0:split], X[split:]
#mean1, mean2 = X1.mean(), X2.mean()
#var1, var2 = X1.var(), X2.var()
#print('mean1=%f, mean2=%f' % (mean1, mean2))
#print('variance1=%f, variance2=%f' % (var1, var2))


##### Show hist(log(X)), first of all we need to remove all value equal 0
#from pandas import Series
#from matplotlib import pyplot
#from numpy import log
#series = Series.from_csv(fullpath + filename, header=0)
#X = series.values
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
from pandas import Series
from statsmodels.tsa.stattools import adfuller
series = Series.from_csv(fullpath + filename, header=0)
X = series.values
result = adfuller(X)        # Ham nay lam tat ca viec kiem dinh
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))



#from pandas import Series
#from statsmodels.tsa.stattools import adfuller
#from numpy import log
#series = Series.from_csv(fullpath + filename, header=0)
#X = series.values
#X = log(X)
#result = adfuller(X)
#print('ADF Statistic: %f' % result[0])
#print('p-value: %f' % result[1])
#for key, value in result[4].items():
#	print('\t%s: %.3f' % (key, value))













