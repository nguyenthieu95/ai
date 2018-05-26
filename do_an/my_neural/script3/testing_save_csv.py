import numpy as np

t1 = [3, 2, 1, 5]
t2 = [4, 6, 0, -1]

t3 = []
t3.append(t1)
t3.append(t2)

t0 = np.reshape(t3, (2, -1))

np.savetxt("test.csv", t0, delimiter=",")



import os, inspect
print(os.path.realpath(__file__))
print os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


print(os.path.dirname(__file__))
print(os.path.abspath(__file__))
print(os.path.basename(__file__))
