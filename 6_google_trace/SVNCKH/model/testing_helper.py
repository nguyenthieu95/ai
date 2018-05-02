from utils import MathHelper
from pandas import read_csv
import cluster

t = MathHelper.distance_func([3, 4], [5, 6])
print("{0}".format(t))

fullpath = "/home/thieunv/university/LabThayMinh/code/data/GoogleTrace/"
filename3 = "data_resource_usage_3Minutes_6176858948.csv"
filename5 = "data_resource_usage_5Minutes_6176858948.csv"
filename8 = "data_resource_usage_8Minutes_6176858948.csv"
filename10 = "data_resource_usage_10Minutes_6176858948.csv"

df = read_csv(fullpath+ filename3, header=None, index_col=False, usecols=[3, 4], engine='python')
dataset_original = df.values
