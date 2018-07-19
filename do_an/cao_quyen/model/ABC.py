import numpy as np
from random import random, randint
from operator import itemgetter
from copy import deepcopy
from sklearn.metrics import mean_absolute_error

class BaseClass(object):
    """
        - Version 1: Copy code trong cuon: Clever Algorithms
    - Cai thien phan create_neigh_bee cho version 0.
    - Ket qua tot hon nhieu. Hoi tu nhanh hon

    - Version ko
    """

    INDEX_BEE = 0
    INDEX_FITNESS = 1

    INDEX_LOWER_VALUE = 0
    INDEX_UPPER_VALUE = 1

    INDEX_BEST_BEE = 0

    def __init__(self, other_para = None, bee_para = None):
        self.number_node_input = other_para["number_node_input"]
        self.number_node_output = other_para["number_node_output"]
        self.X_train = other_para["X_train"]
        self.y_train = other_para["y_train"]
        self.X_valid = other_para["X_valid"]
        self.y_valid = other_para["y_valid"]
        self.activation = other_para["activation"]
        self.train_valid_rate = other_para["train_valid_rate"]

        self.max_gens = bee_para["max_gens"]
        self.num_bees = bee_para["num_bees"]
        self.e_bees = bee_para["couple_num_bees"][0]
        self.o_bees = bee_para["couple_num_bees"][1]

        self.patch_size = bee_para["patch_variables"][0]
        self.patch_factor = bee_para["patch_variables"][1]
        self.num_sites = bee_para["sites"][0]
        self.elite_sites = bee_para["sites"][1]
        self.lowup_w = bee_para["lowup_w"]
        self.lowup_b = bee_para["lowup_b"]

        self.length_matrix_w = self.number_node_input * self.number_node_output
        self.length_vector_b = self.number_node_output
        self.problem_size = self.length_matrix_w + self.length_vector_b
        self.loss_train = []
        self.bee = None


    def create_search_space(self):  # [ [-1, 1], [-1, 1], ... ]
        w2 = [self.lowup_w for i in range(self.length_matrix_w)]
        b2 = [self.lowup_b for i in range(self.length_vector_b)]
        search_space = w2 + b2
        return search_space

    def get_average_mae(self, individual=None, X_data=None, y_data=None):
        w2 = np.reshape(individual[:self.length_matrix_w], (self.number_node_input, -1))
        b2 = np.reshape(individual[self.length_matrix_w:], (-1, self.length_vector_b))
        y_pred = self.activation( np.add(np.matmul(X_data, w2) , b2) )
        return mean_absolute_error(y_pred, y_data)

    def fitness_bee(self, bee=None):
        """ distance between the sum of an indivuduals numbers and the target number. Lower is better"""
        averageTrainMAE = self.get_average_mae(bee, self.X_train, self.y_train)
        averageValidationMAE = self.get_average_mae(bee, self.X_valid, self.y_valid)
        return (self.train_valid_rate[0] * averageTrainMAE + self.train_valid_rate[1] * averageValidationMAE)

    def fitness_encoded(self, encoded):
        return self.fitness_bee(encoded[BaseClass.INDEX_BEE])


    def create_bee(self, minmax=None):
        candidate = [(minmax[i][BaseClass.INDEX_UPPER_VALUE] - minmax[i][BaseClass.INDEX_LOWER_VALUE]) * random() +
                     minmax[i][BaseClass.INDEX_LOWER_VALUE] for i in range(len(minmax))]
        fitness = self.fitness_bee(candidate)
        return [candidate, fitness]


    def create_neigh_bee(self, individual=None, patch_size=None, search_space=None):
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

        fitness = self.fitness_bee(bee)
        return [bee, fitness]


    def search_neigh(self, parent=None, neigh_size=None, search_space=None):  # parent:  [ vector_individual, fitness ]
        """
        Tim kiem trong so neigh_size, chi lay 1 hang xom tot nhat
        """
        neigh = [self.create_neigh_bee(parent[0], self.patch_size, search_space) for _ in range(0, neigh_size)]
        neigh_sorted = sorted(neigh, key=itemgetter(BaseClass.INDEX_FITNESS))
        return deepcopy(neigh_sorted[0])


    def create_scout_bees(self, search_space=None, num_scouts=None):  # So luong ong trinh tham
        return [self.create_bee(search_space) for _ in range(0, num_scouts)]


    def train(self):
        search_space = self.create_search_space()

        pop = [self.create_bee(search_space) for _ in range(0, self.num_bees)]
        best = None
        for j in range(0, self.max_gens):
            pop_sorted = sorted(pop, key=itemgetter(BaseClass.INDEX_FITNESS))
            best = deepcopy(pop_sorted[BaseClass.INDEX_BEST_BEE])

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
            self.loss_train.append(best[BaseClass.INDEX_FITNESS])
            print("Epoch = {0}, patch_size = {1}, Avarage MAE = {2}".format(j + 1, self.patch_size, best[BaseClass.INDEX_FITNESS]))

        print("Train done, fitness = {0}".format(best[BaseClass.INDEX_FITNESS]))
        self.bee = sorted(pop, key=itemgetter(BaseClass.INDEX_FITNESS))[BaseClass.INDEX_BEST_BEE][0]

        return self.bee, self.loss_train




