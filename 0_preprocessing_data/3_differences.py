#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:23:33 2018

@author: thieunv

https://machinelearningmastery.com/difference-time-series-dataset-python/

- Differencing non-stationary data 

"""

fullpath = "/home/thieunv/university/LabThayMinh/code/data/preprocessing/"
filename = "sales-of-shampoo.csv"

from pandas import read_csv
from pandas import datetime, Series
from matplotlib import pyplot

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

series = read_csv(fullpath + filename, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
X = series.values


#### 1. Default dataset 
#pyplot.plot(X)



#### 2. After differencing dataset (Manual by hand)
#diff = difference(X, 4)
#pyplot.plot(diff)



#### 2. After differencing dataset (Auto by pandas)
#diff = series.diff(4)
#pyplot.plot(diff)



#### Show 
pyplot.show()

