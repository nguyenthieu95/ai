#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 01:12:03 2018

@author: thieunv
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 18:08:11 2018

@author: thieunv

- Version 4: Copy code trong cuon:  An improved artificial bee colony algorithm and its application to reliability optimization problems
- Version nay su dung thuat toan cai tien IABC trong paper tren.
- Nhung khi chay thi lai cho ra ket qua toi hon version 3. Chac hon version 2 duoc 1 ti'

"""

from random import random, uniform, randint, sample
from copy import deepcopy
from operator import itemgetter, add
from math import exp, log


def random_vector(minmax):  # minmax: [ [-1, 1], [-1, 1], ... ]
    x = []
    for i in range(len(minmax)):
        x.append((minmax[i][1] - minmax[i][0]) * random() + minmax[i][0])
    return x

def create_food_source(search_space):
    bee = random_vector(search_space)
    fitness = objective_function(bee)
    trial = 0
    return [bee, fitness, trial]

def objective_function(vector):
    return reduce(add, (pow(x, 2.0) for x in vector), 0.0)

def create_neigh_bee_basic(pop, bee, search_space):
    t1 = randint(0, len(bee)-1)
    t2 = randint(0, len(pop)-1)
    new_bee = deepcopy(bee)
    new_bee[t1] = bee[t1] + uniform(-1, 1) * (bee[t1] - pop[t2][0][t1])
    trial = 0
    fitness = objective_function(new_bee)
    return [new_bee, fitness, trial]


def create_neigh_bee_improve(pop, bee, best_global, search_space):
    t1 = randint(0, len(bee)-1)
    t2 = sample(range(0, len(pop)), 2)
    new_bee = deepcopy(bee)
    new_bee[t1] = bee[t1] + uniform(0, 1) * (best_global[t1] - bee[t1] + pop[t2[0]][0][t1] - pop[t2[1]][0][t1])
    trial = 0
    fitness = objective_function(new_bee)
    return [new_bee, fitness, trial]


def abc_algorithm(max_gens=None, SN=None, limit=None, search_space=None, SR=None, SSP_max=None, SSP_min=None, NP_min=None, NP_threshold=None):
    pop = [create_food_source(search_space) for x in range(0, SN)]
    #p_global = None
    pop = sorted(pop, key=itemgetter(1))
    best_global = pop[0]
    iteration = 1
    while(iteration <= max_gens):
        
        if SR > random():
            SN = max(SN-NP_min, NP_threshold)
        
        SSP = SSP_max * exp( iteration* log(SSP_min / SSP_max) / max_gens  )
            
        for i in range(0, SN):
            if random() < SSP:
                new_bee = create_neigh_bee_basic(pop, pop[i][0], search_space)
            else:
                new_bee = create_neigh_bee_improve(pop, pop[i][0], best_global[0], search_space)
                
            if new_bee[1] < pop[i][1]:
                pop[i][0] = new_bee[0]
                pop[i][1] = new_bee[1]
                pop[i][2] = 0
            else:
                pop[i][2] = pop[i][2] + 1
        
        for i in range(0, SN):
            if pop[i][2] > limit:
                pop[i] = create_food_source(search_space)
        
        pop = sorted(pop, key=itemgetter(1))
        best_global = pop[0]
        print("Epoch = {0}, Best is {1}".format(iteration, best_global[1]))
        iteration += 1
    
    return best_global


if __name__ == "__main__":
    #    num_hidden_unit = 8
    #    num_output = 3
    #    problem_size = num_hidden_unit * num_output + num_output    # weights hidden and bias output
    # maximum iteration, SN, limit, SR, ܵܵܲ ௠௔௫, ܵܵܲ௠௜௡ , NPmin, NPThreshold
    
    problem_size = 24
    search_space = [[-1, 1] for i in range(problem_size)]
    
    max_gens = 280  # epoch
    num_bees = 100  # number of bees - population
    limit = 150
    SR = 0.005               # [0.995, 0.7475, 0.5, 0.2525, 0.005]
    NP_min = 5              # [5, 10, 15]
    NP_threshold = 30       # [15, 20, 25]
    SSP_min = 0.1
    SSP_max = 0.8
    

    best = abc_algorithm(max_gens, num_bees, limit, search_space, SR, SSP_max, SSP_min, NP_min, NP_threshold)
    print("done! Solution: f = {0}, s = {1}".format(best[1], best[0]))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
