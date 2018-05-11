#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 01:46:26 2018

@author: thieunv

Artificial Bee Colony algorithm (ABC)
"""

from random import sample, randint
from copy import deepcopy
from operator import itemgetter

class AbcAstimator(object):
    def __init__(self, pop_size=None, p=None, e=None, nep=None, nsp=None):
        self.pop_size = pop_size
        self.p = p
        self.e = e
        self.nep = nep
        self.nsp = nsp
        
        print("Hello world!")
    
    def create_random_bee(self, length_bee=None, minv=-1, maxv=1):
        """
        length_bee: The length of 1 solution
        minv : Min value of each value in solution
        maxv : Max value of each value in solution
        """
        x = (maxv - minv) * np.random.random_sample((length, 1)) + minv 
        fitness = self.fitness_individual(x)
        print("Create random bee")
        return [ x, fitness] 
    
    def initialize_population(self, pop_size=None, length_bee=None, minv=None, maxv=None):
        population = [ self.create_random_bee(length_bee, minv, maxv) for i in range(pop_size) ]
    
    def fitness_individual(self, individual=None):
        print("Tinh fitness")   
    
    def select_individual_for_region(self, pop=None, pop_size=None, p=None, e=None):
        """
        Chọn ngẫu nhiên p vùng từ n vùng ban đầu, tiếp theo là sắp xếp p vùng được chọn này theo
độ thích nghi giảm dần. Khi đó trong p cá thể được sắp thì e cá thể đầu tiên sẽ thuộc e-vùng, p –
e cá thể tiếp theo sẽ thuộc pe-vùng, các cá thể còn lại (chưa được sắp xếp) sẽ thuộc np-vùng.
        """
        pId = sample(range(pop_size), p)
        p_region = [ pop[beeId] for beeId in pId ]
#        p_region_with_fitness = [ (bee, self.fitness_individual(bee)) for bee in p_region ]
#        p_region_with_fitness_sorted = sorted(p_region_with_fitness, key=itemgetter(1))
#        e_region = deepcopy(x[0] for x in p_region_with_fitness_sorted[:e])
#        pe_region = deepcopy(x[0] for x in p_region_with_fitness_sorted[e:])
#        np_region = set(population) - set(e_region) - set(pe_region)
#        return None

    
        p_region_sorted = sorted(p_region, key=itemgetter(1))
        e_region = deepcopy(x for x in p_region_sorted[:e])
        pe_region = deepcopy(x for x in p_region_sorted[e:])
        np_region = set(population) - set(e_region) - set(pe_region)
        return None
    
    def e_sites_neigh_search(self, individual=None, e=None, nep):       # Tìm kiếm lân cận để sinh thêm nep cây khung
        """
        Best region and product more nep bee to neigh search
        """
        for bee in e_region:
            for i in range(0, nep):
                self.find_neigh_region(individual, neighbourhood_individual)
                if self.fitness_individual(neighbourhood_individual) < self.fitness_individual(individual):
                    individual = deepcopy(neighbourhood_individual)
                    
    def pe_sites_neigh_search(self, individual=None, pe=None, nsp):   # Tìm kiếm lân cận để sinh thêm nsp cây khung
        """
        Best region and product more nsp bee to neigh search
        """
        for bee in pe_region:
            for i in range(0, nsp):
                self.find_neigh_region(individual, neighbourhood_individual)
                if self.fitness_individual(neighbourhood_individual) < self.fitness_individual(individual):
                    individual = deepcopy(neighbourhood_individual)
    
        """
        Việc tìm kiếm lân cận T’ cho một cây khung T được thực hiện như sau: Loại ngẫu nhiên khỏi T một cạnh e, 
        sau đó tìm một cạnh e’ tốt nhất từ tập E  T để thay thế và nếu T e + e’ tốt hơn T thì đặt T’ = T e + e’
        """
    def find_neigh(self, indiv=None, neighbour_indiv=None):     
        """
        Xóa ngẫu nhiên một cạnh e thuộc T;
Tìm cạnh e’ tốt nhất trong E  T để thay thế;
        """
        old = deepcopy(indiv)
        indiv[randint(0, len(indiv))] = "best in E-T"
        if self.fitness_individual(indiv) < self.fitness_individual(old):
            indiv = deepcopy(indiv)
        indiv = neighbour_indiv
        
    
    def np_sites_random_search(self, np_region=None):
        """
        Việc tìm kiếm ngẫu nhiên diễn ra ở giai đoạn cuối của mỗi bước lặp khi n  p ong được cử đi
tìm kiếm ngẫu nhiên. Mỗi cá thể trong số np-vùng này được thay thế bằng một cá thể được sinh
ngẫu nhiên mà bỏ qua điều kiện là cá thể ngẫu nhiên này tốt hơn cá thể trước đó ở mỗi vùng đó.
        """
        for bee in np_region:
            bee = create_random_bee()
    
    def find_best_individual(self, pop=None):
        temp = sorted(pop, key=itemgetter(1))
        return temp[0]
    
    
    def train(self):
        pop = self.initialize_population(self.pop_size, self.length_bee, self.minv, self.maxv)
        t = 0
        while t < self.epoch:
            self.select_individual_for_region(pop, self.pop_size, self.p, self.e)
            self.e_sites_neigh_search()
            self.pe_sites_neigh_search()
            self.np_sites_random_search()
            self.update_pop(pop, e_region, pe_region, np_region)
            t += 1
        beebest = self.find_best_individual(pop)
        return beebest
    
    

if __name__ == "__main__":
    pop_size = n = 100
    p = int(2*n/3)
    e = int(p/3)
    nep = 5
    nsp = 3
    minv = -1
    maxv = 1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        




