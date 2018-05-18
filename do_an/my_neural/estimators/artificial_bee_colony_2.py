#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 16:49:54 2018

@author: thieunv
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 01:36:18 2018

@author: thieunv

- Version 2: Copy code trong cuon: Clever Algorithms 
- Cai thien phan create_neigh_bee cho version 1.
- Ket qua tot hon nhieu. Hoi tu nhanh hon

300 epoch --> ver1: 0.8
300 epoch --> ver2: 0.008

"""

from random import random, uniform, randint
from copy import deepcopy
from operator import itemgetter, add


def random_vector(minmax):  # minmax: [ [-1, 1], [-1, 1], ... ]
    x = []
    for i in range(len(minmax)):
        x.append((minmax[i][1] - minmax[i][0]) * random() + minmax[i][0])
    return x


def create_random_bee(search_space):
    return random_vector(search_space)


def objective_function(vector):
    return reduce(add, (pow(x, 2.0) for x in vector), 0.0)


def create_neigh_bee_2(pop, individual, patch_size, search_space):
    t1 = randint(0, len(individual)-1)
    t2 = randint(0, len(pop)-1)
    
    bee = deepcopy(individual)
    bee[t1] = individual[t1] + uniform(-1, 1) * (individual[t1] - pop[t2][0][t1])
    
    if bee[t1] < search_space[i][0]:
        bee[t1] = search_space[i][0]
    if bee[t1] > search_space[i][1]:
        bee[t1] = search_space[i][1]
    return bee


def create_neigh_bee(pop, individual, patch_size, search_space):
    t1 = randint(0, len(individual)-1)
    
    bee = deepcopy(individual)
    if random() < 0.5:
        bee[t1] = individual[t1] + random() * patch_size
    else:
        bee[t1] = individual[t1] - random() * patch_size
            
    if bee[t1] < search_space[t1][0]:
        bee[t1] = search_space[t1][0]
    if bee[t1] > search_space[t1][1]:
        bee[t1] = search_space[t1][1]
    return bee



def search_neigh(pop, parent, neigh_size, patch_size, search_space):  # parent:  [ vector_individual, fitness ]
    """
    Tim kiem trong so neigh_size, chi lay 1 hang xom tot nhat
    """
    neigh = [create_neigh_bee(pop, parent[0], patch_size, search_space) for x in range(0, neigh_size)]
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
            next_gen.append(search_neigh(pop_sorted, pop_sorted[i], neigh_size, patch_size, search_space))

        scouts = create_scout_bees(search_space, (num_bees - num_sites))  # Ong trinh tham
        pop = [x[0] for x in next_gen] + scouts
        patch_size = patch_size * 0.99
        print("Epoch = {0}, patch_size = {1}, best = {2}".format(j + 1, patch_size, best[1]))
    return best



if __name__ == "__main__":
    #    num_hidden_unit = 8
    #    num_output = 3
    #    problem_size = num_hidden_unit * num_output + num_output    # weights hidden and bias output
    problem_size = 24
    search_space = [[-1, 1] for i in range(problem_size)]

    max_gens = 280  # epoch
    num_bees = 100  # number of bees - population
    num_sites = 3  # phan vung, 3 dia diem 
    elite_sites = 1
    patch_size = 3.0
    e_bees = 10
    o_bees = 3

    best = search(max_gens, search_space, num_bees, num_sites, elite_sites, patch_size, e_bees, o_bees)
    print("done! Solution: f = {0}, s = {1}".format(best[1], best[0]))
