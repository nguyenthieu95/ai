import matplotlib.pyplot as plt
import pandas as pd 
from pandas import read_csv
import numpy as np
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error

colnames = ['realData','predict'] 

results_df = read_csv(sys.argv[1], header=None,names=colnames, index_col=False, engine='python')

real = results_df['realData'].values
predictData = results_df['predict'].values

ax = plt.subplot()
ax.plot(real[200:400],label="Actual")
ax.plot(predictData[200:400],label="predictions")
# ax.plrot(TestPred,label="Test")
plt.xlabel("TimeStamp")
plt.ylabel("CPU")

plt.legend()
# plt.savefig('mem5.png')
plt.show()

