#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 22:51:09 2018

@author: thieunv

https://lethain.com/genetic-programming-a-novel-failure/


"""

from random import random, randint



def part():
    options = ('+', '-', '*', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' )
    return options[randint(0, len(options) - 1)]

def individual(length = 5):
    return [ part() for i in range(length) ]

def population(size = 1000, length = 5):
    return [individual(length = length) for x in range(size)]

def fitness(indiv, target):
    try:
        val = eval(" ".join(indiv))
        return target - abs(target - val)
    except:
        return -100000
    
def grade_population_old(pop, target):
    total = sum ( [ fitness(indiv, target) for indiv in pop] )
    avg = total / ( len(pop) * 1.0 )
    return avg

def grade_population(pop, target):
    """ Count the average of number of valids systax in population"""
    fitnesses = [ fitness(indiv, target) for indiv in pop ]
    valids = [ x for x in fitnesses if x > -100000 ]
    return len(valids)



