import numpy as np
import os
from timeit import timeit
from multiprocessing import Pool

def fib(dummy):
    n = [1,1]
    for ii in range(100000):
        n.append(n[-1]+n[-2])

def silly_mult(matrix):
    for row in matrix:
        for val in row:
            val * val

if __name__ == '__main__':

    dt = timeit(lambda: map(fib, range(10)), number=10)
    print("Fibonacci, non-parallel: %.3f" %dt)

    matrices = [np.random.randn(1000,1000) for ii in range(10)]
    dt = timeit(lambda: map(silly_mult, matrices), number=10)
    print("Silly matrix multiplication, non-parallel: %.3f" %dt)

    # after importing numpy, reset the CPU affinity of the parent process so
    # that it will use all CPUS
    os.system("taskset -p 0xff %d" % os.getpid())

    pool = Pool(8)

    dt = timeit(lambda: pool.map(fib,range(10)), number=10)
    print("Fibonacci, parallel: %.3f" %dt)

    dt = timeit(lambda: pool.map(silly_mult, matrices), number=10)
    print("Silly matrix multiplication, parallel: %.3f" %dt)