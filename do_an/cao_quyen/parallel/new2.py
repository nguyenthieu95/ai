from concurrent.futures import ThreadPoolExecutor
import numpy as np


def mmul(matrix):
    for i in range(100):
        matrix = matrix * matrix
    return matrix

matrices = []
for i in range(4):
    matrices.append(np.random.random_integers(100, size=(1000, 1000)))


with ThreadPoolExecutor(max_workers=4) as executor:
    future = executor.map(mmul, matrices)
    print(future)
    