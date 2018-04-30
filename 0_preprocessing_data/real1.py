#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 12:03:34 2018

@author: thieunv
"""


fullpath = "/home/thieunv/university/LabThayMinh/code/data/preprocessing/"
filename = "time_series_cpu.csv"
import pandas as pd
from pandas import Series
from matplotlib import pyplot
from numpy import log
import numpy as np
series = Series.from_csv(fullpath + filename, header=0)
X = series.values


#series.plot()
#pyplot.show()



#dataset = []
#for t in X:
#    if t != 0.0:
#        dataset.append(t)
#X = pd.Series(dataset)
#X = log(X)
pyplot.hist(X)
pyplot.show()




#split = len(X) / 4
#X1, X2, X3, X4 = X[0:split], X[split:2*split], X[2*split:3*split], X[3*split:]
#mean1, mean2, mean3, mean4 = X1.mean(), X2.mean(), X3.mean(), X4.mean()
#var1, var2, var3, var4 = X1.var(), X2.var(), X3.var(), X4.var()
#print('mean1=%f, mean2=%f, mean3=%f, mean4=%f' % (mean1, mean2, mean3, mean4))
#print('variance1=%f, variance2=%f, variance3=%f, variance4=%f' % (var1, var2, var3, var4))



##### The stronger method to check time series using Kiem dinh gia thuyet
import pandas as pd
from pandas import Series
from statsmodels.tsa.stattools import adfuller
series = Series.from_csv(fullpath + filename, header=0)
X = series.values
dataset = []
for t in X:
    if t != 0:
        dataset.append(t)
X = pd.Series(dataset)


result = adfuller(X)        # Ham nay lam tat ca viec kiem dinh
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))








