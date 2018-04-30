#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 22:19:03 2018

@author: thieunv

Genetic Algorithms (GA): https://lethain.com/genetic-algorithms-cool-name-damn-simple/

==== Creating an array of N numbers that equal X when summed ====

- Vong lap co ban cua GA:
    1. Khoi tao dan so
    2. Danh gia ham fitness trung binh cho toan population
    3. Tien hoa
        + Danh gia toan bo population dua vao ham fitness
        + Sap xep tu be den lon 
        + Chon ra parents (20%) cua population
        + Trong 80% con lai lay them 5% de lam parents (Nhu vay no se mang dac tinh phong phu cua dan so)
        + Mutate 1% cua parent (Parents luc nay dang la 25% population)
        + Make children (75%) tu 25% parents
        + Nho la: 1 parent co the lam nhieu` father or mother. Nhung 1 parent khong the la father and mothor cung luc duoc.
        + Ket hop 25% parents va 75% children thanh population moi.

"""


from random import randint, random
from operator import add

def individual(length, min, max):
    return [ randint(min, max) for x in xrange(length) ]

def population(count, length, min, max):
    """
    individual: 1 solution
    count: number of individuals (population)
    length: number of values per individual
    """
    return [ individual(length, min, max) for x in xrange(count) ]

def fitness(individual, target):
    """ distance between the sum of an indivuduals numbers and the target number. Lower is better"""
    sum = reduce(add, individual)  # sum = reduce( (lambda tong, so: tong + so), individual )
    return abs(target - sum)

def grade(pop, target):
    """ Find average fitness for a population"""
    summed = reduce(add, (fitness(indiv, target) for indiv in pop) )
    return summed / (len(pop) * 1.0)

def evolve(pop, target, retain=0.2, random_select=0.05, mutate=0.01):
    graded = [ (fitness(indiv, target), indiv) for indiv in pop ]
    graded = [ x[1] for x in sorted(graded) ]
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]
    
    # Randomly add other individuals to promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)
    
    # Mutate some individuals
    for individual in parents:  # Dot bien cha me (1/100)
        if mutate > random():
            pos_to_mutate = randint(0, len(individual) - 1)
            individual[pos_to_mutate] = randint(min(individual), max(individual))
        
    # Crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length - 1)   # father
        female = randint(0, parents_length - 1) # mother
        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) / 2
            child = male[:half] + female[half:]
            children.append(child)
    
    parents.extend(children)
    return parents


if __name__ == "__main__":
    target = 678            # Gia tri tong
    pop_count = 100         # So luong dan so
    i_length = 4            # 1 ca the co' 5 gia tri
    i_min = 0               # Gia tri nho nhat co the co cua 1 ca the
    i_max = 1000     
    p = population(pop_count, i_length, i_min, i_max)
    fitness_history = [grade(p, target)]
    for i in xrange(2000):
        p = evolve(p, target)
        fitness_history.append(grade(p, target))
    
    for datum in fitness_history:
        print datum














        
    