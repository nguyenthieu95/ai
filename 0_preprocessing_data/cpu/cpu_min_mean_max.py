#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:50:53 2018

@author: thieunv
"""

fullpath = "/home/thieunv/university/LabThayMinh/code/data/preprocessing/"
filename = "time_series_cpu.csv"

from pandas import Series, DataFrame, concat
from sklearn import preprocessing

series = Series.from_csv(fullpath + filename, header=0)
dataset_transform = preprocessing.MinMaxScaler().fit_transform(series.values.reshape(-1, 1))

#### 3. Rolling Window Statistics
# Su dung moi~ 1 ham thong ke thanh 1 feature moi (dua vao cac gia tri truoc do)

## 1 ham thong ke: min, mean, max

temps = DataFrame(series.values)
width = 3
temps = DataFrame(dataset_transform)
shifted = temps.shift(width - 1)    # Shift 1 gia tri de co cho~ ma` thay the cho mean features
window = shifted.rolling(window=width)   # Dua vao 2 gia tri truoc do de tinh mean, sau do thay the vao cho gia tri shift
means = window.mean()
dataframe = concat([window.min(), window.mean(), window.max(), temps], axis=1)
dataframe.columns = ['min', 'mean', 'max', 't+1']
print(dataframe)