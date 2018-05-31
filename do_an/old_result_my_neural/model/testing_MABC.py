import numpy as np
from random import random, randint
from operator import add

SN = 10                                     # Number of Employed Bees
n = 10                                      # Number of Flowers in Each Food Source
X = np.zeros((SN, n))                                  # Food Source
V = np.zeros((SN, n))                                  # New Food Source for Worker Bee
U = np.zeros((SN, n))                                  # New Food Source for Onlooker Bee

Trial = np.zeros(SN)                  # Counter for Abandonment
Cycle = 1
Max_Cycle = 180
Best_ABC = np.zeros( (Max_Cycle, Max_Cycle))	# Best Solution
Sum = np.zeros(SN)
p = np.zeros(SN)                          # Probability of each Food Source
Pre_Val = 0.5                               # Pre Determined Value For Selecting Food Source by Onlooker Bees
Max_Trial = 30                              # Maximum Number of Abandonment Trial
Sol = np.zeros(SN)                    # Solution Value
Max_Iteration = 1
Result = np.zeros((Max_Iteration , Max_Cycle))

## Problem Definition: For Initialization in MABC Algorithm
Max_K = SN
ch = np.zeros((Max_K , n))
O_X = [SN, n]             	                # Opposition of Food Source
X_Min = np.zeros((n, n))
X_Max = np.zeros((n, n))

num = input('Write the Number of Function: ')

### ABC Algorithm
# Initialization of the Population
for i in range(0, SN):
    for j in range(0, n):
        X[i][j] = j + random() * (n - j)



while Cycle <= Max_Cycle:   # Produce a New Food Source Population for Employed Bees
    for i in range(0, SN):
        Flag = False        # Improve in Food Resource Checker
        for j in range(0, n):

            phi = -1 + 2 * random()
            k = randint(1, 10)

            if k == i:
                k = randint(1, 10)

            V[i][j] = X[i][j] + phi * (X[i][j] - X[k][j])

        for j in range(0, n):
            if X[i][j] <= V[i][j]:
                X[i][j] = V[i][j]
                Flag = True
        if Flag == True:
            Trial[i] = 0
            Flag = False
        else:
            Trial[i] += 1


    const = 1100

    # Calculating Probability Values
    for i in range(0, SN):
        for j in range(0, n):
            Sum[i] += const
    total = 0
    for i in range(0, SN):
        total += Sum[i]

    total = reduce(add, Sum, 0)
    p[i] = Sum[i] / total

    # Produce a New Food Source Population for Onlooker Bees
    t = 0
    i = 1
    C = 0
    while (t <= SN and C < Max_Cycle):
        if p[i] > Pre_Val:
            Flag = False                # Improve in Food Resource Checker
            for j in range(0, n):
                phi = -1 + 2 * random()
                k = randint(1, 10)
                if k == i:
                    k = randint(1, 10)
                U[i][j] = X[i][j] + phi * (X[i][j] - X[k][j])

            for j in range(0, n):
                if X[i][j] < U[i][j]:
                    X[i][j] = U[i][j]
                    Flag = True

            if Flag == True:
                Trial[i] = 0
                Flag = False
            else:
                Trial[i] += 1

            t += 1

        C = C + 1
        i = i + 1
        if i > SN:
            i = 1

    # Determine Scout Bees
    if max(Trial[i] > Max_Trial):
        for j in range(0, n):
            X[i][j] = j + random() * (n-j)

    const = 100
    # Determine the Best Solution
    for i in range(0, SN):
        for j in range(0, n):
            Sol[i] = Sol[i] + const

    total = reduce(add, Sol, 0)
    Sol[i] = Sol[i] / total

    for i in range(0, SN):
        if Best_ABC[Cycle] < Sol[i]:
            Best_ABC[Cycle] = Sol[i]

    print('Cycle & BestSolution of ABS is: {0}, {1} '.format(Cycle, Best_ABC[Cycle]))
    Cycle = Cycle + 1

