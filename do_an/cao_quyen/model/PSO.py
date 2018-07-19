import numpy as np
from random import random
from copy import deepcopy
from sklearn.metrics import mean_absolute_error

class BaseClass(object):
    INDEX_CURRENT_POSITION = 0
    INDEX_PAST_POSITION_BEST = 1
    INDEX_VECTOR_V = 2
    INDEX_CURRENT_FITNESS = 3
    INDEX_PAST_FITNESS = 4

    INDEX_LOWER_VALUE = 0
    INDEX_UPPER_VALUE = 1

    def __init__(self, other_para = None, pso_para = None):
        self.number_node_input = other_para["number_node_input"]
        self.number_node_output = other_para["number_node_output"]
        self.X_train = other_para["X_train"]
        self.y_train = other_para["y_train"]
        self.X_valid = other_para["X_valid"]
        self.y_valid = other_para["y_valid"]
        self.activation = other_para["activation"]
        self.train_valid_rate = other_para["train_valid_rate"]

        self.epoch =  pso_para["epoch"]
        self.pop_size = pso_para["pop_size"]
        self.c1 = pso_para["c_couple"][0]
        self.c2 = pso_para["c_couple"][1]
        self.w_min = pso_para["w_minmax"][0]
        self.w_max = pso_para["w_minmax"][1]
        self.lowup_w =  pso_para["lowup_w"]
        self.lowup_b = pso_para["lowup_b"]

        self.length_matrix_w = self.number_node_input * self.number_node_output
        self.length_vector_b = self.number_node_output
        self.problem_size = self.length_matrix_w + self.length_vector_b
        self.loss_train = []


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

    def create_individual(self, minmax=None):
        """
                x: vi tri hien tai cua con chim
                x_past_best: vi tri trong qua khu ma` ga`n voi thuc an (best result) nhat
                v: vector van toc cua con chim (cung so chieu vs x)
        """
        x = np.reshape([(minmax[i][BaseClass.INDEX_UPPER_VALUE] - minmax[i][BaseClass.INDEX_LOWER_VALUE]) * random() +
             minmax[i][BaseClass.INDEX_LOWER_VALUE] for i in range(len(minmax))], (-1, 1))
        x_past_best = deepcopy(x)
        v = np.zeros((len(x), 1))
        x_fitness = self.fitness_individual(x)
        x_past_fitness = deepcopy(x_fitness)
        return [x, x_past_best, v, x_fitness, x_past_fitness]


    def get_global_best(self, pop):
        sorted_pop = sorted(pop, key=lambda temp: temp[BaseClass.INDEX_CURRENT_FITNESS])
        return deepcopy(sorted_pop[0])

    def fitness_individual(self, individual):
        """ distance between the sum of an indivuduals numbers and the target number. Lower is better"""
        averageTrainMAE = self.get_average_mae(individual, self.X_train, self.y_train)
        averageValidationMAE = self.get_average_mae(individual, self.X_valid, self.y_valid)
        return (self.train_valid_rate[0] * averageTrainMAE + self.train_valid_rate[1] * averageValidationMAE)

    def fitness_particle(self, particle):
        return self.fitness_individual(particle[BaseClass.INDEX_CURRENT_POSITION])


    def train(self):
        """
        - Khoi tao quan the (tinh ca global best)
        - Di chuyen va update vi tri, update gbest
        """
        self.search_space = self.create_search_space()
        pop = [self.create_individual(self.search_space) for _ in range(self.pop_size)]

        gbest = self.get_global_best(pop)
        self.loss_train.append(gbest[BaseClass.INDEX_CURRENT_FITNESS])

        for i in range(self.epoch):
            # Update weight after each move count  (weight down)
            w = (self.epoch - i) / self.epoch * (self.w_max - self.w_min) + self.w_min
            for j in range(self.pop_size):
                r1 = np.random.random_sample()
                r2 = np.random.random_sample()
                #%% testing
                vi_sau = w * pop[j][BaseClass.INDEX_VECTOR_V] + self.c1 * r1 * (pop[j][BaseClass.INDEX_PAST_POSITION_BEST] - pop[j][BaseClass.INDEX_CURRENT_POSITION]) \
                         + self.c2 * r2 * (gbest[BaseClass.INDEX_CURRENT_POSITION] - pop[j][BaseClass.INDEX_CURRENT_POSITION])

                #%% testing 2
                xi_sau = pop[j][BaseClass.INDEX_CURRENT_POSITION] + vi_sau                 # Xi(sau) = Xi(truoc) + Vi(sau) * deltaT (deltaT = 1)
                fit_sau = self.fitness_individual(xi_sau)
                fit_truoc = pop[j][BaseClass.INDEX_PAST_FITNESS]

                # Cap nhat x hien tai, v hien tai, so sanh va cap nhat x past best voi x hien tai
                pop[j][BaseClass.INDEX_CURRENT_POSITION] = deepcopy(xi_sau)
                pop[j][BaseClass.INDEX_VECTOR_V] = deepcopy(vi_sau)
                pop[j][BaseClass.INDEX_CURRENT_FITNESS] = fit_sau

                if fit_sau < fit_truoc:
                    pop[j][BaseClass.INDEX_PAST_POSITION_BEST] = deepcopy(xi_sau)
                    pop[j][BaseClass.INDEX_PAST_FITNESS] = fit_sau

            gbest = self.get_global_best(pop)
            self.loss_train.append(gbest[BaseClass.INDEX_CURRENT_FITNESS])
            print("Generation : {0}, average MAE over population: {1}".format(i+1, gbest[BaseClass.INDEX_CURRENT_FITNESS]))

        self.individual, self.loss_train = gbest[BaseClass.INDEX_CURRENT_POSITION], self.loss_train[1:]
        print("Build model and train done!!!")

        return self.individual, self.loss_train



