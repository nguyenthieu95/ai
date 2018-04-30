import pandas as pd
import numpy as np
import os
from pandas import read_csv


df = read_csv('/home/thieunv/university/LabThayMinh/code/6_google_trace/data/resource_usage_twoMinutes_6176858948.csv', header=None,index_col=False)

df1 = df.replace(np.nan, 0, regex=True)
df1.to_csv('/home/thieunv/university/LabThayMinh/code/6_google_trace/data/twoMinutes_6176858948_notNan.csv', index=False, header=None)
# print df1
