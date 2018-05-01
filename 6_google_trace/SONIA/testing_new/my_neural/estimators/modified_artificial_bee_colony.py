#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 17:12:00 2018

@author: thieunv

THE ARTIFICIAL BEE COLONY ALGORITHM
IN TRAINING ARTIFICIAL NEURAL
NETWORK FOR OIL SPILL DETECTION

- Kha la hieu qua neu problem size lon
"""

from random import random
from copy import deepcopy
from operator import itemgetter, add


def initial_population(minmax=None, N=None):
    """
    N: population size
    """
    pop = []
    for i in range(N):
        xi = []
        oxi = []
        for j in range(len(minmax)):
            temp = (minmax[j][1] - minmax[j][0]) * random() + minmax[j][0]
            xi.append(temp)
            oxi.append(minmax[j][1] + minmax[j][0] - temp)
        pop.append(xi)
        pop.append(oxi)
    pop_temp = [(xij, objective_function(xij)) for xij in pop ]
    pop_sorted = sorted(pop_temp, key=itemgetter(1))
    del pop_temp
    return pop_sorted[:N]

def stype_subpopulation(pop=None, N=None, M=None):
    """
    M: number of sub group
    """
    s = int(N/M)
    group = []
    for i in range(N):
        
        temp = []
        for j in range(s):
    
    
    
    
        

def search_main():
    N = 80
    M = 3
    max_iteration = 1500
    
    pop = initial_population(minmax, N)
    stype_subpopulation(pop, N, M)
    trial = 0   # 1 < trial < food_number
    # initialize elitists to be food source - chon ra moi~ vung trong M vu`ng 1 con tot nhat (M-bees)
    best_food_source = find_best(M-bees)
    
    Iter = 0
    while Iter <= max_iteration:
        


def objective_function(vector):
    return reduce(add, (pow(x, 2.0) for x in vector), 0.0)



def random_vector(minmax):  # minmax: [ [-1, 1], [-1, 1], ... ]
    x = []
    for i in range(len(minmax)):
        x.append((minmax[i][1] - minmax[i][0]) * random() + minmax[i][0])
    return x



def create_random_bee(search_space):
    return random_vector(search_space)




def create_neigh_bee(individual, patch_size, search_space):
    bee = []
    elem = 0.0
    for x in range(0, len(individual)):
        if random() < 0.5:
            elem = individual[x] + random() * patch_size
        else:
            elem = individual[x] - random() * patch_size

        if elem < search_space[i][0]:
            elem = search_space[i][0]
        if elem > search_space[i][1]:
            elem = search_space[i][1]
        bee.append(deepcopy(elem))
    return bee


def search_neigh(parent, neigh_size, patch_size, search_space):  # parent:  [ vector_individual, fitness ]
    """
    Tim kiem trong so neigh_size, chi lay 1 hang xom tot nhat
    """
    neigh = [create_neigh_bee(parent[0], patch_size, search_space) for x in range(0, neigh_size)]
    neigh = [(bee, objective_function(bee)) for bee in neigh]
    neigh_sorted = sorted(neigh, key=itemgetter(1))
    return neigh_sorted[0]


def create_scout_bees(search_space, num_scouts):  # So luong ong trinh tham
    return [create_random_bee(search_space) for x in range(0, num_scouts)]


def search(max_gens, search_space, num_bees, num_sites, elite_sites, patch_size, e_bees, o_bees):
    pop = [create_random_bee(search_space) for x in range(0, num_bees)]
    for j in range(0, max_gens):
        pop = [(bee, objective_function(bee)) for bee in pop]
        pop_sorted = sorted(pop, key=itemgetter(1))
        best = pop_sorted[0]

        next_gen = []
        for i in range(0, num_sites):
            if i < elite_sites:
                neigh_size = e_bees
            else:
                neigh_size = o_bees
            next_gen.append(search_neigh(pop_sorted[i], neigh_size, patch_size, search_space))

        scouts = create_scout_bees(search_space, (num_bees - num_sites))  # Ong trinh tham
        pop = [x[0] for x in next_gen] + scouts
        patch_size = patch_size * 0.99
        print("Epoch = {0}, patch_size = {1}, best = {2}".format(j + 1, patch_size, best[1]))
    return best


if __name__ == "__main__":
    #    num_hidden_unit = 8
    #    num_output = 3
    #    problem_size = num_hidden_unit * num_output + num_output    # weights hidden and bias output
    problem_size = 60
    search_space = [[-1, 1] for i in range(problem_size)]
    
    food_number = CS / 2    # CS: colony size
    

    max_gens = 1500  # epoch
    num_bees = 100  # number of bees - population
    num_sites = 3  # phan vung, 3 dia diem 
    elite_sites = 1
    patch_size = 3.0
    e_bees = 7
    o_bees = 2

    best = search(max_gens, search_space, num_bees, num_sites, elite_sites, patch_size, e_bees, o_bees)
    print("done! Solution: f = {0}, s = {1}".format(best[1], best[0]))
