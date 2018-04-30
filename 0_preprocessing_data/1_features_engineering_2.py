#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:13:32 2018

@author: thieunv
"""

fullpath = "/home/thieunv/university/LabThayMinh/code/data/preprocessing/"
filename = "daily-minimum-temperatures-in-me.csv"

from pandas import Series
from pandas import DataFrame
from pandas import concat
series = Series.from_csv(fullpath + filename, header=0)


#### 3. Rolling Window Statistics
# Su dung moi~ 1 ham thong ke thanh 1 feature moi (dua vao cac gia tri truoc do)

## 1 ham thong ke: mean
#temps = DataFrame(series.values)
#shifted = temps.shift(1)    # Shift 1 gia tri de co cho~ ma` thay the cho mean features
#window = shifted.rolling(window=2)   # Dua vao 2 gia tri truoc do de tinh mean, sau do thay the vao cho gia tri shift
#means = window.mean()
#dataframe = concat([means, temps], axis=1)
#dataframe.columns = ['mean(t-2,t-1)', 't+1']
#print(dataframe.head(10))



## 3 ham thong ke: min, mean, max
temps = DataFrame(series.values)
width = 3
shifted = temps.shift(width - 1)
window = shifted.rolling(window=width)
dataframe = concat([window.min(), window.mean(), window.max(), temps], axis=1)
dataframe.columns = ['min', 'mean', 'max', 't+1']
print(dataframe.head(10))



#### 4. Expanding Window Statistics
#temps = DataFrame(series.values)
#window = temps.expanding()
#dataframe = concat([window.min(), window.mean(), window.max(), temps.shift(-1)], axis=1)
#dataframe.columns = ['min', 'mean', 'max', 't+1']
#print(dataframe.head(15))












