import matplotlib.pyplot as plt
from pandas import read_csv
import sys

colnames = ['realData','predict']
results_df = read_csv(sys.argv[1], header=None,names=colnames, index_col=False, engine='python')

real = results_df['realData'].values
predictData = results_df['predict'].values

ax = plt.subplot()
ax.plot(real[0:150],label="Actual")
ax.plot(predictData[0:150],label="Predict")
plt.xlabel("Time (5 minutes)")
plt.ylabel("Memory Usage")
ax.set_title('FLNN')

plt.legend()
plt.savefig('top2.pdf')
plt.show()

