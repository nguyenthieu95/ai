#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 14:16:49 2018

@author: thieunv
"""

fullpath = "/home/thieunv/university/LabThayMinh/code/data/preprocessing/"
filename = "time_series_cpu.csv"

from pandas import Series, DataFrame, concat, read_csv
from sklearn import preprocessing
import numpy as np

#series = Series.from_csv(fullpath + filename, header=0)
#dataset_transform = preprocessing.MinMaxScaler().fit_transform(series.values.reshape(-1, 1))

df = read_csv(fullpath+ filename, header=0, index_col=False, usecols=[1], engine='python', dtype=np.float64)   
dataset_original = df.values
dataset_transform = preprocessing.MinMaxScaler().fit_transform(dataset_original)

#### 3. Rolling Window Statistics
# Su dung moi~ 1 ham thong ke thanh 1 feature moi (dua vao cac gia tri truoc do)

## 1 ham thong ke: mean
temps = DataFrame(dataset_transform)
shifted = temps.shift(1)    # Shift 1 gia tri de co cho~ ma` thay the cho mean features
window = shifted.rolling(window=3)   # Dua vao 2 gia tri truoc do de tinh mean, sau do thay the vao cho gia tri shift
means = window.mean()
dataframe = concat([means, temps], axis=1)
dataframe.columns = ['mean(t-2,t-1)', 't+1']
print(dataframe)











