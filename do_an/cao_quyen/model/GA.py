import numpy as np
from random import random
from copy import deepcopy
from operator import itemgetter
from sklearn.metrics import mean_absolute_error

class BaseClass(object):
    FITNESS_INDEX_SORTED = 1            # 0: Chromosome, 1: fitness (so choice 1)
    FITNESS_INDEX_AFTER_SORTED = -1     # High fitness choose, 0: when low fitness choose
    """
    Link:
        https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9
        https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_quick_guide.htm
        https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/
    """
    def __init__(self, other_para = None, ga_para = None):
        self.number_node_input = other_para["number_node_input"]
        self.number_node_output = other_para["number_node_output"]
        self.X_train = other_para["X_train"]
        self.y_train = other_para["y_train"]
        self.X_valid = other_para["X_valid"]
        self.y_valid = other_para["y_valid"]
        self.activation = other_para["activation"]

        self.lowup_w =  ga_para["lowup_w"]
        self.lowup_b = ga_para["lowup_b"]
        self.epoch = ga_para["epoch"]
        self.pop_size = ga_para["pop_size"]
        self.pc = ga_para["pc"]
        self.pm = ga_para["pm"]

        self.length_matrix_w = self.number_node_input * self.number_node_output
        self.length_vector_b = self.number_node_output
        self.problem_size = self.length_matrix_w + self.length_vector_b
        self.loss_train = []

    def create_search_space(self):  # [ [-1, 1], [-1, 1], ... ]
        w2 = [self.lowup_w for i in range(self.length_matrix_w)]
        b2 = [self.lowup_b for i in range(self.length_vector_b)]
        search_space = w2 + b2
        return search_space


    def create_chromosome(self, minmax=None):
        chromosome = [(minmax[i][1] - minmax[i][0]) * random() + minmax[i][0] for i in range(len(minmax))]
        fitness = self.compute_fitness(chromosome, self.X_train, self.y_train)
        return [chromosome, fitness]


    def compute_fitness(self, chromosome=None, X_data=None, y_data=None):
        w2 = np.reshape(chromosome[:self.length_matrix_w], (self.number_node_input, -1))
        b2 = np.reshape(chromosome[self.length_matrix_w:], (-1, self.length_vector_b))
        y_pred = self.activation( np.add( np.matmul(X_data, w2), b2) )
        return 1.0 / mean_absolute_error(y_pred, y_data)



    ### Selection
    def get_index_roulette_wheel_selection(self, list_fitness, sum_fitness):
        r = np.random.uniform(low=0, high=sum_fitness)
        for idx, f in enumerate(list_fitness):
            r = r + f
            if r > sum_fitness:
                return idx

    def get_index_tournament_selection(self, pop=None, k_way=10):
        random_selected = np.random.choice(range(0, self.problem_size), k_way, replace=False)
        temp = [pop[i] for i in random_selected]
        temp = sorted(temp, key=itemgetter(1))
        return temp[BaseClass.FITNESS_INDEX_AFTER_SORTED]

    def get_index_stochastic_universal_selection(self, list_fitness, sum_fitness):
        r = np.random.uniform(low=0, high=sum_fitness)
        round1, round2 = r, r
        selected = []
        time1, time2 = False, False

        for idx, f in enumerate(list_fitness):
            round1 = round1 + f
            round2 = round2 - f
            if time1 and time2:
                break
            if not time1:
                if round1 > sum_fitness:
                    selected.append(idx)
                    time1 = True
            if not time2:
                if round2 < 0:
                    selected.append(idx)
                    time2 = True
        return selected


    ### Crossover
    def crossover_one_point(self, dad=None, mom=None):
        point = np.random.randint(0, len(dad))
        w1 = dad[:point] + mom[point:]
        w2 = mom[:point] + dad[point:]
        return w1, w2

    def crossover_multi_point(self, dad=None, mom=None):
        r = np.random.choice(range(0, len(dad)), 2, replace=False)
        a, b = min(r), max(r)
        w1 = dad[:a] + mom[a:b] + dad[b:]
        w2 = mom[:a] + dad[a:b] + mom[b:]
        return w1, w2

    def crossover_uniform(self, dad=None, mom=None):
        r = np.random.uniform()
        w1, w2 = deepcopy(dad), deepcopy(mom)
        for i in range(0, len(dad)):
            if np.random.uniform() < 0.7:   # bias to the dad   (equal when 0.5)
                w1[i] = dad[i]
                w2[i] = mom[i]
            else:
                w1[i] = mom[i]
                w2[i] = dad[i]
        return w1, w2

    def crossover_arthmetic_recombination(self, dad=None, mom=None):
        r = np.random.uniform()             # w1 = w2 when r =0.5
        w1 = np.multiply(r, dad) + np.multiply((1 - r), mom)
        w2 = np.multiply(r, mom) + np.multiply((1 - r), dad)
        return w1, w2


    ### Mutation
    def mutation_flip_point(self, parent):
        point = np.random.randint(0, len(parent))
        w = deepcopy(parent)
        w[point] = np.random.uniform(self.search_space[point][0], self.search_space[point][1])
        return w

    def mutation_swap(self, parent):
        r = np.random.choice(range(0, len(parent)), 2, replace=False)
        w = deepcopy(parent)
        w[r[0]], w[r[1]] = w[r[1]], w[r[0]]
        return w

    def mutation_scramble(self, parent):
        r = np.random.choice(range(0, len(parent)), 2, replace=False)
        a, b = min(r), max(r)


    ### Survivor Selection
    def survivor_gready(self, pop_old=None, pop_new=None):
        pop = [ pop_new[i] if pop_new[i][1] > pop_old[i][1] else pop_old[i] for i in range(self.pop_size)]
        return pop

    def create_next_generation(self, pop):
        next_population = []

        list_fitness = [pop[i][1] for i in range(self.pop_size)]
        fitness_sum = sum(list_fitness)
        while (len(next_population) < self.pop_size):

            ### Selection
            c1 = pop[self.get_index_roulette_wheel_selection(list_fitness, fitness_sum)]
            c2 = pop[self.get_index_roulette_wheel_selection(list_fitness, fitness_sum)]

            w1 = deepcopy(c1[0])
            w2 = deepcopy(c2[0])

            ### Crossover
            if np.random.uniform() < self.pc:
                w1, w2 = self.crossover_arthmetic_recombination(c1[0], c2[0])

            ### Mutation
            if np.random.uniform() < self.pm:
                w1 = self.mutation_flip_point(w1)
            if np.random.uniform() < self.pm:
                w2 = self.mutation_flip_point(w2)

            c1_new = [w1, self.compute_fitness(w1, self.X_train, self.y_train)]
            c2_new = [w2, self.compute_fitness(w2, self.X_train, self.y_train)]
            next_population.append(c1_new)
            next_population.append(c2_new)
        return next_population


    def train(self):
        best_chromosome_train = None
        best_fitness_train = -1
        self.search_space = self.create_search_space()
        pop = [ self.create_chromosome(self.search_space) for _ in range(self.pop_size) ]

        for j in range(0, self.epoch):
            # Next generations
            pop = deepcopy(self.create_next_generation(pop))

            # Find best chromosome
            pop_sorted = sorted(pop, key=itemgetter(BaseClass.FITNESS_INDEX_SORTED))
            best_chromosome_train = deepcopy(pop_sorted[BaseClass.FITNESS_INDEX_AFTER_SORTED])

            if best_chromosome_train[1] > best_fitness_train:
                best_fitness_train = best_chromosome_train[1]
            # print("> Epoch {0}: Best training fitness {1}".format(j + 1, 1.0 / best_fitness_train))
            self.loss_train.append(1.0 / best_chromosome_train[1])

        #print("done! Solution: f = {0}, MAE = {1}".format(best_chromosome_train[0], 1.0 / best_chromosome_train[1]))
        return best_chromosome_train[0], self.loss_train



class Ver1(BaseClass):
    """
    Using survival gready selection.
    Ket qua kem hon so voi BaseClass
    """

    def __init__(self, other_para=None, ga_para=None):
        super().__init__(other_para, ga_para)

    def create_next_generation(self, pop):
        next_population = []

        list_fitness = [pop[i][1] for i in range(self.pop_size)]
        fitness_sum = sum(list_fitness)
        while (len(next_population) < self.pop_size):

            ### Selection
            c1 = pop[self.get_index_roulette_wheel_selection(list_fitness, fitness_sum)]
            c2 = pop[self.get_index_roulette_wheel_selection(list_fitness, fitness_sum)]

            w1 = deepcopy(c1[0])
            w2 = deepcopy(c2[0])

            ### Crossover
            if np.random.uniform() < self.pc:
                w1, w2 = self.crossover_arthmetic_recombination(c1[0], c2[0])

            ### Mutation
            if np.random.uniform() < self.pm:
                w1 = self.mutation_flip_point(w1)
            if np.random.uniform() < self.pm:
                w2 = self.mutation_flip_point(w2)

            c1_new = [w1, self.compute_fitness(w1, self.X_train, self.y_train)]
            c2_new = [w2, self.compute_fitness(w2, self.X_train, self.y_train)]
            next_population.append(c1_new)
            next_population.append(c2_new)

        next_population = super().survivor_gready(pop, next_population)
        return next_population


