from utils.PreprocessingUtil import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.MathUtil import *
from utils.IOUtil import *
from random import random
from copy import deepcopy
from operator import itemgetter

class Model:
    FITNESS_INDEX_SORTED = 1  # 0: Chromosome, 1: fitness (so choice 1)
    FITNESS_INDEX_AFTER_SORTED = -1  # High fitness choose, 0: when low fitness choose

    def __init__(self, dataset_original=None, train_idx=None, valid_idx=None, test_idx=None, sliding=None, activation=None,
                 expand_func=None, epoch=None, pop_size=None, pc=None, pm=None, lowup_w=(-1, 1), lowup_b=(-1, 1),
                 test_name=None, path_save_result=None, method_statistic = None, output_index=None, output_multi=None):
        self.data_original = dataset_original
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        if valid_idx == 0:
            self.valid_idx = 0
        else:
            self.valid_idx = int(train_idx + (test_idx - train_idx) / 2)

        self.sliding = sliding
        self.scaler = MinMaxScaler()
        self.expand_func = expand_func
        self.activation = activation
        self.epoch = epoch
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm
        self.lowup_w = lowup_w
        self.lowup_b = lowup_b
        self.output_index = output_index
        self.method_statistic = method_statistic
        self.path_save_result = path_save_result
        self.test_name = test_name
        self.output_multi = output_multi
        self.filename = "FL_GANN-sliding_{0}-ex_func_{1}-act_func_{2}-epoch_{3}-pop_size_{4}-pc_{5}-pm_{6}".format(
            sliding,expand_func, activation,epoch,pop_size, pc, pm)

        if activation == 0:
            self.activation_function = itself
        elif activation == 1:
            self.activation_function = elu
        elif activation == 2:
            self.activation_function = relu
        elif activation == 3:
            self.activation_function = tanh
        elif activation == 4:
            self.activation_function = sigmoid

        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = None, None, None, None, None, None
        self.chromosome, self.loss_train = None, None
        self.number_node_input, self.number_node_output = None, None
        self.length_matrix_w, self.length_vector_b, self.problem_size = None, None, None


    def preprocessing_data(self):
        timeseries = TimeSeries(self.expand_func, self.train_idx, self.valid_idx, self.test_idx, self.sliding, self.method_statistic, self.data_original, self.scaler)
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, self.scaler = timeseries.preprocessing(self.output_index)
        #print("Processing data done!!!")

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

    ### Crossover
    def crossover_arthmetic_recombination(self, dad=None, mom=None):
        r = np.random.uniform()  # w1 = w2 when r =0.5
        w1 = np.multiply(r, dad) + np.multiply((1 - r), mom)
        w2 = np.multiply(r, mom) + np.multiply((1 - r), dad)
        return w1, w2

    ### Mutation
    def mutation_flip_point(self, parent):
        point = np.random.randint(0, len(parent))
        w = deepcopy(parent)
        w[point] = np.random.uniform(self.search_space[point][0], self.search_space[point][1])
        return w

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
            pop_sorted = sorted(pop, key=itemgetter(Model.FITNESS_INDEX_SORTED))
            best_chromosome_train = deepcopy(pop_sorted[Model.FITNESS_INDEX_AFTER_SORTED])

            if best_chromosome_train[1] > best_fitness_train:
                best_fitness_train = best_chromosome_train[1]
            # print("> Epoch {0}: Best training fitness {1}".format(j + 1, 1.0 / best_fitness_train))
            self.loss_train.append(1.0 / best_chromosome_train[1])

        #print("done! Solution: f = {0}, MAE = {1}".format(best_chromosome_train[0], 1.0 / best_chromosome_train[1]))
        return best_chromosome_train[0], self.loss_train


    def predict(self):
        w = np.reshape(self.chromosome[:self.length_matrix_w], (self.number_node_input, self.number_node_output))
        b = np.reshape(self.chromosome[self.length_matrix_w:], (-1, self.number_node_output))
        y_pred = self.activation_function( np.add(np.matmul(self.X_test, w), b) )

        self.pred_inverse = self.scaler.inverse_transform(y_pred)
        self.real_inverse = self.scaler.inverse_transform(self.y_test)

        self.mae = np.round(mean_absolute_error(self.pred_inverse, self.real_inverse, multioutput='raw_values'), 4)
        self.rmse = np.round(np.sqrt(mean_squared_error(self.pred_inverse, self.real_inverse, multioutput='raw_values')), 4)

        if self.output_multi:
            write_all_results([self.filename, self.rmse[0], self.rmse[1], self.mae[0], self.mae[1] ], self.test_name, self.path_save_result)
            save_result_to_csv(self.real_inverse[:,0:1], self.pred_inverse[:,0:1], self.filename, self.path_save_result+"CPU-")
            save_result_to_csv(self.real_inverse[:,1:2], self.pred_inverse[:,1:2], self.filename, self.path_save_result+"RAM-")
            #draw_predict_with_error(1, self.real_inverse[:,0:1], self.pred_inverse[:,0:1], self.rmse[0], self.mae[0], self.filename, self.path_save_result+"CPU-")
            #draw_predict_with_error(2, self.real_inverse[:,1:2], self.pred_inverse[:,1:2], self.rmse[1], self.mae[1], self.filename, self.path_save_result+"RAM-")
        else:
            write_all_results([self.filename, self.rmse[0], self.mae[0] ], self.test_name, self.path_save_result)
            save_result_to_csv(self.real_inverse, self.pred_inverse, self.filename, self.path_save_result)
            #draw_predict_with_error(1, self.real_inverse, self.pred_inverse, self.rmse[0], self.mae[0], self.filename, self.path_save_result)

    def run(self):
        self.preprocessing_data()
        self.number_node_input = self.X_train.shape[1]
        self.number_node_output = self.y_train.shape[1]
        self.length_matrix_w = self.number_node_input * self.number_node_output
        self.length_vector_b = self.number_node_output
        self.problem_size = self.length_matrix_w + self.length_vector_b
        self.chromosome, self.loss_train = self.train()
        self.predict()



