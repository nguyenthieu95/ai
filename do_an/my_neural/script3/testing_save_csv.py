import numpy as np

t1 = [3, 2, 1, 5]
t2 = [4, 6, 0, -1]

t3 = []
t3.append(t1)
t3.append(t2)

t0 = np.reshape(t3, (2, -1))

np.savetxt("test.csv", t0, delimiter=",")
