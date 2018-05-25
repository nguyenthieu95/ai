import numpy as np
from random import random, randint, uniform, sample
from math import exp, log
from copy import deepcopy
from operator import itemgetter
from sklearn.metrics import mean_absolute_error

class Bee1(object):
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
            # print("Epoch = {0}, patch_size = {1}, best = {2}".format(j + 1, self.patch_size, best[1]))
        return best


    def build_and_train(self):
        search_space = self.create_search_space()
        best = self.search(search_space)
        # print("done! Solution: f = {0}, s = {1}".format(best[1], best[0]))
        self.bee = best[0]

        return self.bee, self.loss_train





class Bee2(object):
    """
    - Version 3: Copy code trong cuon:  An improved artificial bee colony algorithm and its application to reliability optimization problems
- Thi voi ABC basic neu ra trong bai bao do. Neu su dung ham` xac suat. if random() < pop[i][1] / p_global:
- Thi ket qua lai toi. Chac chi ngang bang voi version 1

- Gio neu bo ha`m xac suat di. Chi ca`n su dung random() < 0.5 thi ket qua la tot nhat trong 4 version

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
        self.limit = bee_para["limit"]
        self.size_w2 = self.number_node_input * self.number_node_output
        self.size_b2 = self.number_node_output

        self.loss_train = []

    def create_search_space(self):  # [ [-1, 1], [-1, 1], ... ]
        w2 = [self.lowup_w for i in range(self.size_w2)]
        b2 = [self.lowup_b for i in range(self.size_b2)]
        search_space = w2 + b2
        return search_space

    def random_vector(self, minmax=None):  # minmax: [ [-1, 1], [-1, 1], ... ]
        x = []
        for i in range(len(minmax)):
            x.append((minmax[i][1] - minmax[i][0]) * random() + minmax[i][0])
        return x

    def create_food_source(self, search_space=None):
        bee = self.random_vector(search_space)
        fitness = self.objective_function(bee)
        trial = 0
        return [bee, fitness, trial]

    def objective_function(self, vector=None):
        return self.get_mae(vector)


    def get_mae(self, bee=None):
        w2 = np.reshape(bee[:self.size_w2], (self.number_node_input, -1))
        b2 = np.reshape(bee[self.size_w2:], (-1, self.size_b2))
        output = np.add( np.matmul(self.X_data, w2), b2)
        y_pred = self.activation(output)
        return mean_absolute_error(y_pred, self.y_data)


    def create_neigh_bee(self, pop=None, bee=None):
        t1 = randint(0, len(bee) - 1)
        t2 = randint(0, len(pop) - 1)
        new_bee = deepcopy(bee)
        new_bee[t1] = bee[t1] + uniform(-1, 1) * (bee[t1] - pop[t2][0][t1])
        trial = 0
        fitness = self.objective_function(new_bee)
        return [new_bee, fitness, trial]



    def search(self, search_space=None):
        pop = [self.create_food_source(search_space) for x in range(0, self.num_bees)]
        #p_global = None
        best_global = None
        iteration = 0
        while (iteration < self.max_gens):

            for i in range(0, self.num_bees):
                new_bee = self.create_neigh_bee(pop, pop[i][0])
                if new_bee[1] < pop[i][1]:
                    pop[i][0] = new_bee[0]
                    pop[i][1] = new_bee[1]
                    pop[i][2] = 0
                else:
                    pop[i][2] = pop[i][2] + 1
            #p_global = reduce(add, (bee[1] for bee in pop), 0.0)
            for i in range(0, self.num_bees):
                # if random() < pop[i][1] / p_global:
                if random() < 0.5:
                    new_bee = self.create_neigh_bee(pop, pop[i][0])
                    if new_bee[1] < pop[i][1]:
                        pop[i][0] = new_bee[0]
                        pop[i][1] = new_bee[1]
                        pop[i][2] = 0
                    else:
                        pop[i][2] = pop[i][2] + 1

            for i in range(0, self.num_bees):
                if pop[i][2] > self.limit:
                    pop[i] = self.create_food_source(search_space)

            pop = sorted(pop, key=itemgetter(1))
            best_global = pop[0]
            self.loss_train.append(best_global[1])
            # print("Epoch = {0}, Best is {1}".format(iteration, best_global[1]))
            iteration += 1

        return best_global


    def build_and_train(self):
        search_space = self.create_search_space()
        best = self.search(search_space)
        # print("done! Solution: f = {0}, s = {1}".format(best[1], best[0]))
        self.bee = best[0]

        return self.bee, self.loss_train





class Bee3(object):
    """
- Version 4: Copy code trong cuon:  An improved artificial bee colony algorithm and its application to reliability optimization problems
- Version nay su dung thuat toan cai tien IABC trong paper tren.
- Nhung khi chay thi lai cho ra ket qua toi hon version 3. Chac hon version 2 duoc 1 ti'
    """
    def __init__(self, other_para = None, bee_para = None):
        """
        :param max_gens:
        :param max_gens: pop size
        :param limit: limit number of trial, after limit if bee not improve --> it turn into scout bee
        :param search_space:
        :param SR: Survival rate.
        :param SSP_max: Selection scheme the parameter
        :param SSP_min:
        :param NP_min: pop size nho nhat co the tru di
        :param NP_threshold: pop size nguong
        :return:
        """
        self.number_node_input = other_para["number_node_input"]
        self.number_node_output = other_para["number_node_output"]
        self.X_data = other_para["X_data"]
        self.y_data = other_para["y_data"]
        self.activation = other_para["activation"]

        self.lowup_w =  bee_para["lowup_w"]
        self.lowup_b = bee_para["lowup_b"]
        self.max_gens = bee_para["max_gens"]
        self.num_bees = bee_para["num_bees"]
        self.limit = bee_para["limit"]
        self.SR = bee_para["SR"]
        self.NP_min = bee_para["NP_min"]
        self.NP_threshold = bee_para["NP_threshold"]
        self.SSP_min = bee_para["SSP_min"]
        self.SSP_max = bee_para["SSP_max"]
        self.size_w2 = self.number_node_input * self.number_node_output
        self.size_b2 = self.number_node_output

        self.loss_train = []

    def create_search_space(self):  # [ [-1, 1], [-1, 1], ... ]
        w2 = [self.lowup_w for i in range(self.size_w2)]
        b2 = [self.lowup_b for i in range(self.size_b2)]
        search_space = w2 + b2
        return search_space

    def random_vector(self, minmax=None):  # minmax: [ [-1, 1], [-1, 1], ... ]
        x = []
        for i in range(len(minmax)):
            x.append((minmax[i][1] - minmax[i][0]) * random() + minmax[i][0])
        return x

    def create_food_source(self, search_space=None):
        bee = self.random_vector(search_space)
        fitness = self.objective_function(bee)
        trial = 0
        return [bee, fitness, trial]

    def objective_function(self, vector=None):
        return self.get_mae(vector)

    def get_mae(self, bee=None):
        w2 = np.reshape(bee[:self.size_w2], (self.number_node_input, -1))
        b2 = np.reshape(bee[self.size_w2:], (-1, self.size_b2))
        output = np.add( np.matmul(self.X_data, w2), b2)
        y_pred = self.activation(output)
        return mean_absolute_error(y_pred, self.y_data)

    def create_neigh_bee_basic(self, pop=None, bee=None):
        t1 = randint(0, len(bee) - 1)
        t2 = randint(0, len(pop) - 1)
        new_bee = deepcopy(bee)
        new_bee[t1] = bee[t1] + uniform(-1, 1) * (bee[t1] - pop[t2][0][t1])
        trial = 0
        fitness = self.objective_function(new_bee)
        return [new_bee, fitness, trial]

    def create_neigh_bee_improve(self, pop=None, bee=None, best_global=None):
        t1 = randint(0, len(bee) - 1)
        t2 = sample(range(0, len(pop)), 2)
        new_bee = deepcopy(bee)
        new_bee[t1] = bee[t1] + uniform(0, 1) * (best_global[t1] - bee[t1] + pop[t2[0]][0][t1] - pop[t2[1]][0][t1])
        trial = 0
        fitness = self.objective_function(new_bee)
        return [new_bee, fitness, trial]

    def search(self, search_space=None):
        pop = [self.create_food_source(search_space) for x in range(0, self.num_bees)]
        pop = sorted(pop, key=itemgetter(1))
        best_global = pop[0]
        iteration = 1
        while (iteration <= self.max_gens):

            if self.SR > random():
                SN = max(self.num_bees - self.NP_min, self.NP_threshold)

            SSP = self.SSP_max * exp(iteration * log(self.SSP_min / self.SSP_max) / self.max_gens)

            for i in range(0, self.num_bees):
                if random() < SSP:
                    new_bee = self.create_neigh_bee_basic(pop, pop[i][0])
                else:
                    new_bee = self.create_neigh_bee_improve(pop, pop[i][0], best_global[0])

                if new_bee[1] < pop[i][1]:
                    pop[i][0] = new_bee[0]
                    pop[i][1] = new_bee[1]
                    pop[i][2] = 0
                else:
                    pop[i][2] = pop[i][2] + 1

            for i in range(0, self.num_bees):
                if pop[i][2] > self.limit:
                    pop[i] = self.create_food_source(search_space)

            pop = sorted(pop, key=itemgetter(1))
            best_global = pop[0]
            self.loss_train.append(best_global[1])
            # print("Epoch = {0}, Best is {1}".format(iteration, best_global[1]))
            iteration += 1

        return best_global

    def build_and_train(self):
        search_space = self.create_search_space()
        best = self.search(search_space)
        # print("done! Solution: f = {0}, s = {1}".format(best[1], best[0]))
        self.bee = best[0]

        return self.bee, self.loss_train


