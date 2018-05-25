#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 16:19:55 2018

@author: thieunv
"""
from preprocessing import TimeSeries
from cluster import Clustering
from utils import MathHelper, GraphUtil, IOHelper

from math import sqrt
import numpy as np
from copy import deepcopy
from operator import add
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing

class Model(object):
    def __init__(self, dataset_original=None, list_idx=(1000, 2000, 1), output_index=0, sliding=2, method_statistic=0, max_cluster=15,
                 positive_number=0.15, sti_level=0.15, dis_level=0.25, mutation_id=1, couple_acti=(2, 0), fig_id=0, pathsave=None,
                 max_move=100, pop_size=45, c_couple=(1.2, 1.2), w_minmax=(0.4, 0.9), value_minmax=(-1, +1)):
        self.dataset_original = dataset_original
        self.output_index = output_index
        self.sliding = sliding
        self.max_cluster = max_cluster
        self.method_statistic = method_statistic
        self.stimulation_level = sti_level
        self.distance_level = dis_level
        self.positive_number = positive_number
        self.mutation_id = mutation_id
        self.pathsave = pathsave
        self.fig_id = fig_id
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.standard_scaler = preprocessing.StandardScaler()

        self.max_move = max_move
        self.pop_size = pop_size
        self.c1 = c_couple[0]
        self.c2 = c_couple[1]
        self.w_min = w_minmax[0]
        self.w_max = w_minmax[1]
        self.value_min = value_minmax[0]
        self.value_max = value_minmax[1]
        self.activation_id1 = couple_acti[0]
        if couple_acti[0] == 0:
            self.activation1 = MathHelper.elu
        elif couple_acti[0] == 1:
            self.activation1 = MathHelper.relu
        elif couple_acti[0] == 2:
            self.activation1 = MathHelper.tanh
        else:
            self.activation1 = MathHelper.sigmoid

        if couple_acti[1] == 0:
            self.activation2 = MathHelper.elu
        elif couple_acti[1] == 1:
            self.activation2 = MathHelper.relu
        elif couple_acti[1] == 2:
            self.activation2 = MathHelper.tanh
        else:
            self.activation2 = MathHelper.sigmoid

        self.train_idx = list_idx[0]
        self.test_idx = list_idx[1]
        if list_idx[2] == 0:
            self.valid_idx = 0
        else:
            self.valid_idx = int(list_idx[0] + (list_idx[1] - list_idx[0]) / 2)

        self.filename = 'Slid=' + str(sliding) + '_PN=' + str(positive_number) + '_SL=' + str(sti_level) + '_DL=' + str(dis_level) + '_MM=' + str(max_move) + '_PS=' + str(pop_size) \
            + '_c1=' + str(c_couple[0]) + '_c2=' + str(c_couple[1])

    def preprocessing_data(self):
        timeseries = TimeSeries(self.train_idx, self.valid_idx, self.test_idx, self.sliding, self.method_statistic, self.dataset_original, self.min_max_scaler)
        if self.valid_idx == 0:
            self.X_train, self.y_train, self.X_test, self.y_test, self.min_max_scaler = timeseries.net_single_output(self.output_index)
        else:
            self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, self.min_max_scaler = timeseries.net_single_output(self.output_index)
        # print("Processing data done!!!")


    def clustering_data(self):
        self.clustering = Clustering(stimulation_level=self.stimulation_level, positive_number=self.positive_number, max_cluster=self.max_cluster,
                                distance_level=self.distance_level, mutation_id=self.mutation_id, activation_id=self.activation_id1, dataset=self.X_train)
        self.centers, self.list_clusters, self.count_centers = self.clustering.sobee_with_mutation()
        # print("Encoder features done!!!")

    def transform_data(self):
        self.S_train = self.clustering.transform_features(self.X_train)
        self.S_test = self.clustering.transform_features(self.X_test)
        if self.valid_idx != 0:
            self.S_valid = self.clustering.transform_features(self.X_valid)
        # print("Transform features done!!!")



    def get_average_mae(self, weightHO, X_data, y_data):
        mae_list = []
        for i in range(0, len(X_data)):
            y_out = np.dot(X_data[i], weightHO)
            mae_list.append(abs(self.activation2(y_out) - y_data[i]))
        temp = reduce(add, mae_list, 0)
        return temp / (len(X_data) * 1.0)

    def grade_pop(self, pop):
        """ Find average fitness for a population"""
        summed = reduce(add, (self.fitness_encode(indiv) for indiv in pop))
        return summed / (len(pop) * 1.0)

    def get_global_best(self, pop):
        sorted_pop = sorted(pop, key=lambda temp: temp[3])
        gbest = deepcopy([sorted_pop[0][0], sorted_pop[0][3]])
        return gbest

    def individual(self, length, min=-1, max=1):
        """
        x: vi tri hien tai cua con chim
        x_past_best: vi tri trong qua khu ma` ga`n voi thuc an (best result) nhat
        v: vector van toc cua con chim (cung so chieu vs x)
        """
        x = (max - min) * np.random.random_sample((length, 1)) + min
        x_past_best = deepcopy(x)
        v = np.zeros((x.shape[0], 1))
        x_fitness = self.fitness_individual(x)
        x_past_fitness = deepcopy(x_fitness)
        return [x, x_past_best, v, x_fitness, x_past_fitness]

    def population(self, pop_size, length, min=-1, max=1):
        """
        individual: 1 solution
        pop_size: number of individuals (population)
        length: number of values per individual
        """
        return [self.individual(length, min, max) for x in range(pop_size)]

    def fitness_encode(self, encode):
        """ distance between the sum of an indivuduals numbers and the target number. Lower is better"""
        averageTrainMAE = self.get_average_mae(encode[0], self.S_train, self.y_train)
        averageValidationMAE = self.get_average_mae(encode[0], self.S_valid, self.y_valid)
        #        sum = reduce(add, individual)  # sum = reduce( (lambda tong, so: tong + so), individual )
        return 0.4 * averageTrainMAE + 0.6 * averageValidationMAE

    def fitness_individual(self, individual):
        """ distance between the sum of an indivuduals numbers and the target number. Lower is better"""
        averageTrainMAE = self.get_average_mae(individual, self.S_train, self.y_train)
        averageValidationMAE = self.get_average_mae(individual, self.S_valid, self.y_valid)
        return (0.4 * averageTrainMAE + 0.6 * averageValidationMAE)

    def build_model_and_train(self):
        """
        - Khoi tao quan the (tinh ca global best)
        - Di chuyen va update vi tri, update gbest
        """
        pop = self.population(self.pop_size, len(self.list_clusters), self.value_min, self.value_max)
        fitness_history = []

        gbest = self.get_global_best(pop)
        fitness_history.append(gbest[1])

        for i in range(self.max_move):
            # Update weight after each move count  (weight down)
            w = (self.max_move - i) / self.max_move * (self.w_max - self.w_min) + self.w_min

            for j in range(self.pop_size):
                r1 = np.random.random_sample()
                r2 = np.random.random_sample()
                vi_sau = w * pop[j][2] + self.c1 * r1 * (pop[j][1] - pop[j][0]) + self.c2 * r2 * (
                            gbest[0] - pop[j][0])
                xi_sau = pop[j][0] + vi_sau  # Xi(sau) = Xi(truoc) + Vi(sau) * deltaT (deltaT = 1)
                fit_sau = self.fitness_individual(xi_sau)
                fit_truoc = pop[j][4]
                # Cap nhat x hien tai, v hien tai, so sanh va cap nhat x past best voi x hien tai
                pop[j][0] = deepcopy(xi_sau)
                pop[j][2] = deepcopy(vi_sau)
                pop[j][3] = fit_sau

                if fit_sau < fit_truoc:
                    pop[j][1] = deepcopy(xi_sau)
                    pop[j][4] = fit_sau

            gbest = self.get_global_best(pop)
            fitness_history.append(gbest[1])
            # print "Generation : {0}, average MAE over population: {1}".format(i+1, gbest[1])

        self.weight, self.loss_train = gbest[0], fitness_history[1:]
    # print("Build model and train done!!!")


    def predict(self):

        w2 = self.weight
        y_pred = []
        for i in range(0, len(self.S_test)):
            output = np.matmul(self.S_test[i], w2).reshape(-1, 1)
            out = np.apply_along_axis(self.activation2, 1, output)
            y_pred.append(out.flatten())

        # Evaluate models on the test set
        y_test_inverse = self.min_max_scaler.inverse_transform(self.y_test)
        y_pred_inverse = self.min_max_scaler.inverse_transform(np.array(y_pred).reshape(-1, self.y_test.shape[1]))

        testScoreRMSE = sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
        testScoreMAE = mean_absolute_error(y_test_inverse, y_pred_inverse)

        self.y_predict, self.score_test_RMSE, self.score_test_MAE = y_pred, testScoreRMSE, testScoreMAE
        self.y_test_inverse, self.y_pred_inverse = y_test_inverse, y_pred_inverse

        # print('DONE - RMSE: %.5f, MAE: %.5f' % (testScoreRMSE, testScoreMAE))
        # print("Predict done!!!")


    def draw_result(self):
        GraphUtil.draw_loss(self.fig_id, self.max_move, self.loss_train, "Loss on training per epoch")
        GraphUtil.draw_predict_with_mae(self.fig_id+1, self.y_test_inverse, self.y_pred_inverse, self.score_test_RMSE,
                                        self.score_test_MAE, "Model predict", self.filename, self.pathsave)

    def save_result(self):
        IOHelper.save_result_to_csv(self.y_test_inverse, self.y_pred_inverse, self.filename, self.pathsave)

    def fit(self):
        self.preprocessing_data()
        self.clustering_data()
        if self.count_centers <= self.max_cluster:
            self.transform_data()
            self.build_model_and_train()
            self.predict()
            self.draw_result()
            self.save_result()


