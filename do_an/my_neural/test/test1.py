import os
from pandas import read_csv

fullpath = os.path.abspath('../../data')

filename = "/data_resource_usage_3Minutes_6176858948.csv"

print(fullpath + filename)

df = read_csv(fullpath+ filename, header=None, index_col=False, usecols=[3], engine='python')
dataset_original = df.values