import numpy as np
from random import random, randint
from copy import deepcopy
from operator import itemgetter
from sklearn.metrics import mean_absolute_error

class Bee(object):
    """
    - Version 1: Copy code trong cuon: Clever Algorithms
- Cai thien phan create_neigh_bee cho version 0.
- Ket qua tot hon nhieu. Hoi tu nhanh hon

- Version ko
    """
    def __init__(self, other_para = None, bee_para = None):
        self.number_node_input = other_para["number_node_input"]
        self.number_node_output = other_para["number_node_output"]
        self.X_data = other_para["X_data"]
        self.y_data = other_para["y_data"]
        self.activation = other_para["activation"]

        self.lowup_w =  bee_para["lowup_w"]
        self.lowup_b = bee_para["lowup_b"]
        self.max_gens = bee_para["max_gens"]
        self.num_bees = bee_para["num_bees"]
        self.num_sites = bee_para["num_sites"]
        self.elite_sites = bee_para["elite_sites"]
        self.patch_size = bee_para["patch_size"]
        self.patch_factor = bee_para["patch_factor"]
        self.e_bees = bee_para["e_bees"]
        self.o_bees = bee_para["o_bees"]
        self.size_w2 = self.number_node_input * self.number_node_output
        self.size_b2 = self.number_node_output

        self.loss_train = []

    def create_search_space(self):  # [ [-1, 1], [-1, 1], ... ]
        w2 = [self.lowup_w for i in range(self.size_w2)]
        b2 = [self.lowup_b for i in range(self.size_b2)]
        search_space = w2 + b2
        return search_space

    def create_candidate(self, minmax=None):
        candidate = [(minmax[i][1] - minmax[i][0]) * random() + minmax[i][0] for i in range(len(minmax))]
        fitness = self.get_mae(candidate)
        return [candidate, fitness]

    def get_mae(self, bee=None):
        w2 = np.reshape(bee[:self.size_w2], (self.number_node_input, -1))
        b2 = np.reshape(bee[self.size_w2:], (-1, self.size_b2))
        output = np.add( np.matmul(self.X_data, w2), b2)
        y_pred = self.activation(output)
        return mean_absolute_error(y_pred, self.y_data)

    def create_random_bee(self, search_space):
        return self.create_candidate(search_space)

    def objective_function(self, vector):
        return self.get_mae(vector)

    def create_neigh_bee_version_0(self, individual=None, patch_size=None, search_space=None):
        bee = []
        for x in range(0, len(individual)):
            elem = 0.0
            if random() < 0.5:
                elem = individual[x] + random() * patch_size
            else:
                elem = individual[x] - random() * patch_size

            if elem < search_space[x][0]:
                elem = search_space[x][0]
            if elem > search_space[x][1]:
                elem = search_space[x][1]
            bee.append(deepcopy(elem))
        return bee


    def create_neigh_bee_version_1(self, individual, patch_size, search_space):
        t1 = randint(0, len(individual) - 1)

        bee = deepcopy(individual)
        if random() < 0.5:
            bee[t1] = individual[t1] + random() * patch_size
        else:
            bee[t1] = individual[t1] - random() * patch_size

        if bee[t1] < search_space[t1][0]:
            bee[t1] = search_space[t1][0]
        if bee[t1] > search_space[t1][1]:
            bee[t1] = search_space[t1][1]

        fitness = self.get_mae(bee)
        return [bee, fitness]


    def search_neigh(self, parent, neigh_size, search_space):  # parent:  [ vector_individual, fitness ]
        """
        Tim kiem trong so neigh_size, chi lay 1 hang xom tot nhat
        """
        neigh = [self.create_neigh_bee_version_1(parent[0], self.patch_size, search_space) for x in range(0, neigh_size)]
        neigh_sorted = sorted(neigh, key=itemgetter(1))
        return neigh_sorted[0]


    def create_scout_bees(self, search_space, num_scouts):  # So luong ong trinh tham
        return [self.create_random_bee(search_space) for x in range(0, num_scouts)]


    def search(self, search_space):
        pop = [self.create_random_bee(search_space) for x in range(0, self.num_bees)]
        best = None
        for j in range(0, self.max_gens):
            pop_sorted = sorted(pop, key=itemgetter(1))
            best = pop_sorted[0]

            next_gen = []
            for i in range(0, self.num_sites):
                if i < self.elite_sites:
                    neigh_size = self.e_bees
                else:
                    neigh_size = self.o_bees
                next_gen.append(self.search_neigh(pop_sorted[i], neigh_size, search_space))

            scouts = self.create_scout_bees(search_space, (self.num_bees - self.num_sites))  # Ong trinh tham
            pop = next_gen + scouts
            self.patch_size = self.patch_size * self.patch_factor
            self.loss_train.append(best[1])
            #print("Epoch = {0}, patch_size = {1}, best = {2}".format(j + 1, self.patch_size, best[1]))
        return best


    def build_and_train(self):
        search_space = self.create_search_space()
        best = self.search(search_space)
        #print("done! Solution: f = {0}, s = {1}".format(best[1], best[0]))
        self.bee = best[0]

        return self.bee, self.loss_train

