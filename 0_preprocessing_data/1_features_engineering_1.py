#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 16:41:39 2018

@author: thieunv

https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/

Daily minimum temperatures in Melbourne, Australia, 1981-1990 (3,650 observations)

"""


fullpath = "/home/thieunv/university/LabThayMinh/code/data/preprocessing/"
filename = "daily-minimum-temperatures-in-me.csv"

from pandas import Series
from pandas import DataFrame
from pandas import concat
series = Series.from_csv(fullpath + filename, header=0)


dataframe = DataFrame()
dataframe['month'] = [series.index[i].month for i in range(len(series))]
dataframe['day'] = [series.index[i].day for i in range(len(series))]
dataframe['temperature'] = [series[i] for i in range(len(series))]
print(dataframe)

#### 1. Date Time Features
# Ta co the trich xuat ra nhieu cau hoi de co the che' them du lieu
#Minutes elapsed for the day.
#Hour of day.
#Business hours or not.
#Weekend or not.
#Season of the year.
#Business quarter of the year.
#Daylight savings or not.
#Public holiday or not.
#Leap year or not.


#### 2. Lag Features (width = 1)
temps = DataFrame(series.values)
dataframe = concat([temps.shift(1), temps], axis=1)
dataframe.columns = ['t-1', 't+1']
print(dataframe)

#### 2. Lag Features (width = 3)
temps = DataFrame(series.values)
dataframe = concat([temps.shift(3), temps.shift(2), temps.shift(1), temps], axis=1)
dataframe.columns = ['t-3', 't-2', 't-1', 't+1']
print(dataframe.head(10))



























