#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 11:49:54 2018

@author: thieunv 

https://machinelearningmastery.com/time-series-data-stationary-python/

https://www.quora.com/What-is-non-stationary-data

https://stats.stackexchange.com/questions/2077/how-to-make-a-time-series-stationary


- Stational data
"""

fullpath = "/home/thieunv/university/LabThayMinh/code/data/preprocessing/"
filename = "daily-total-female-births-in-cal.csv"


#### Show point
from pandas import Series
from matplotlib import pyplot
series = Series.from_csv(fullpath + filename, header=0)
series.plot()
pyplot.show()


#### Show distribution (basiclly normal distribution)
from pandas import Series
from matplotlib import pyplot
series = Series.from_csv(fullpath + filename, header=0)
series.hist()
pyplot.show()


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



##### The stronger method to check time series using Kiem dinh gia thuyet
#from pandas import Series
#from statsmodels.tsa.stattools import adfuller
#series = Series.from_csv(fullpath + filename, header=0)
#X = series.values
#result = adfuller(X)        # Ham nay lam tat ca viec kiem dinh
#print('ADF Statistic: %f' % result[0])
#print('p-value: %f' % result[1])
#print('Critical Values:')
#for key, value in result[4].items():
#	print('\t%s: %.3f' % (key, value))



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




