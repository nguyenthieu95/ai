#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 18:08:11 2018

@author: thieunv

- Version 3: Copy code trong cuon:  An improved artificial bee colony algorithm and its application to reliability optimization problems
- Thi voi ABC basic neu ra trong bai bao do. Neu su dung ham` xac suat. if random() < pop[i][1] / p_global:
- Thi ket qua lai toi. Chac chi ngang bang voi version 2

- Gio neu bo ha`m xac suat di. Chi ca`n su dung random() < 0.5 thi ket qua la tot nhat trong 4 version

"""

from random import random, uniform, randint
from copy import deepcopy
from operator import itemgetter, add


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


def create_neigh_bee(pop, bee, search_space):
    t1 = randint(0, len(bee)-1)
    t2 = randint(0, len(pop)-1)
    new_bee = deepcopy(bee)
    new_bee[t1] = bee[t1] + uniform(-1, 1) * (bee[t1] - pop[t2][0][t1])
    trial = 0
    fitness = objective_function(new_bee)
    return [new_bee, fitness, trial]


def abc_algorithm(max_gens=None, SN=None, limit=None, search_space=None):
    pop = [create_food_source(search_space) for x in range(0, SN)]
    p_global = None
    best_global = None
    iteration = 0
    while(iteration < max_gens):
        
        for i in range(0, SN):
            new_bee = create_neigh_bee(pop, pop[i][0], search_space)
            if new_bee[1] < pop[i][1]:
                pop[i][0] = new_bee[0]
                pop[i][1] = new_bee[1]
                pop[i][2] = 0
            else:
                pop[i][2] = pop[i][2] + 1
        p_global = reduce(add, (bee[1] for bee in pop), 0.0)
        
        for i in range(0, SN):
            #if random() < pop[i][1] / p_global:
            if random() < 0.5:   
                new_bee = create_neigh_bee(pop, pop[i][0], search_space)
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
    problem_size = 24
    search_space = [[-1, 1] for i in range(problem_size)]
    limit = 150
    max_gens = 280  # epoch
    num_bees = 100  # number of bees - population

    best = abc_algorithm(max_gens, num_bees, limit, search_space)
    print("done! Solution: f = {0}, s = {1}".format(best[1], best[0]))
