#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 10:47:41 2018

@author: thieunv

- PSO update gbest sau khi di chuyen toan bo ca dan chim (chuan)

- Multivariate - Cluster + Mutation + PSO 

- 2input -> 1 output

- Cluster dung he so: 2param

"""

import tensorflow as tf
import numpy as np
from scipy.spatial import distance
from math import exp, sqrt
import copy
from random import randint, uniform 
from operator import itemgetter, add
from pandas import read_csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Model(object):
    def __init__(self, dataset_original, list_idx, pso_param, sliding, positive_number, sti_level, dis_level = 0.25, method_statistic = 0, activation=0, d_couple=(0.5, 0.5)):
        self.dataset_original = dataset_original
        self.train_idx = list_idx[0]
        self.validation_idx = int(list_idx[0] + (list_idx[1] - list_idx[0])/2)
        self.test_idx = list_idx[1]
        self.pop_size = pso_param[0]
        self.move_count = pso_param[1]
        self.value_min = pso_param[2]
        self.value_max = pso_param[3]
        self.w_min = pso_param[4]
        self.w_max = pso_param[5]
        self.c1 = pso_param[6]
        self.c2 = pso_param[7]     
        self.sliding = sliding
        self.positive_number = positive_number
        self.stimulation_level = sti_level
        self.distance_level = dis_level
        self.method_statistic = method_statistic
        self.activation = activation
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.standard_scaler = preprocessing.StandardScaler()
        self.dimension = dataset_original.shape[1]
        self.alpha = d_couple[0]
        self.beta = d_couple[1]
        self.filenamesave = "point-slid_{0}-method_{1}-pos_{2}-sti_{3}-dis_{4}-pop_size_{5}-move_count_{6}-c1_{7}-c2_{8}-activ_{9}-alpha_{10}-beta_{11}".format(sliding, method_statistic, positive_number, sti_level, dis_level, pso_param[0], pso_param[1], pso_param[6], pso_param[7], activation, d_couple[0], d_couple[1])
        
    def preprocessing_data(self):
        """
            cpu(t), cpu(t-1), ..., ram(t), ram(t-1),... 
        """
        train_idx, validation_idx, test_idx, dataset_original, sliding = self.train_idx, self.validation_idx, self.test_idx, self.dataset_original, self.sliding
        ## Transform all dataset
        list_split = []     # ram, disk, cpu, 
        for i in range(self.dimension-1, -1, -1):
            list_split.append(dataset_original[:test_idx + sliding, i:i+1])
            
        list_transform = []     # ram, disk, cpu
        for i in range(self.dimension):
            list_transform.append(self.min_max_scaler.fit_transform(list_split[i]))
        
        ## Handle data with sliding
        dataset_sliding = np.zeros(shape=(test_idx,1))
        for i in range(self.dimension-1, -1, -1):
            for j in range(sliding):
                d1 = np.array(list_transform[i][j: test_idx+j])
                dataset_sliding = np.concatenate((dataset_sliding, d1), axis=1)
        dataset_sliding = dataset_sliding[:, 1:]
#        print("done")
            
        ## window value: x1 \ x2 \ x3  (dataset_sliding)
        ## Now we using different method on this window value 
        dataset_y = copy.deepcopy(list_transform[self.dimension-1][sliding:])      # Now we need to find dataset_X
        
        if self.method_statistic == 0:
            dataset_X = copy.deepcopy(dataset_sliding)

        if self.method_statistic == 1:
            """
            mean(x1, x2, x3, ...), mean(t1, t2, t3,...) x: cpu, t: ram
            """
            xTB = np.reshape(np.mean(dataset_sliding[:, 0:sliding], axis = 1), (-1, 1))
            tTB = np.reshape(np.mean(dataset_sliding[:, sliding:], axis = 1), (-1, 1))
            dataset_X = np.concatenate( (xTB, tTB), axis=1 )
            
        if self.method_statistic == 2:
            """
            min(x1, x2, x3, ...), mean(x1, x2, x3, ...), max(x1, x2, x3, ....)
            """
            min_X = np.reshape(np.amin(dataset_sliding[:, 0:sliding], axis = 1), (-1, 1))
            mean_X = np.reshape(np.mean(dataset_sliding[:, 0:sliding], axis = 1), (-1, 1))
            max_X = np.reshape(np.amax(dataset_sliding[:, 0:sliding], axis = 1), (-1, 1))
            
            min_T = np.reshape(np.amin(dataset_sliding[:, sliding:], axis = 1), (-1, 1))
            mean_T = np.reshape(np.mean(dataset_sliding[:, sliding:], axis = 1), (-1, 1))
            max_T = np.reshape(np.amax(dataset_sliding[:, sliding:], axis = 1), (-1, 1))
            
            dataset_X = np.concatenate( (min_X, mean_X, max_X, min_T, mean_T, max_T), axis=1 )
            
        if self.method_statistic == 3:
            """
            min(x1, x2, x3, ...), median(x1, x2, x3, ...), max(x1, x2, x3, ....)
            """
            min_X = np.reshape(np.amin(dataset_sliding[:, 0:sliding], axis = 1), (-1, 1))
            median_X = np.reshape(np.median(dataset_sliding[:, 0:sliding], axis = 1), (-1, 1))
            max_X = np.reshape(np.amax(dataset_sliding[:, 0:sliding], axis = 1), (-1, 1))
            
            min_T = np.reshape(np.amin(dataset_sliding[:, sliding:], axis = 1), (-1, 1))
            median_T = np.reshape(np.median(dataset_sliding[:, sliding:], axis = 1), (-1, 1))
            max_T = np.reshape(np.amax(dataset_sliding[:, sliding:], axis = 1), (-1, 1))
            
            dataset_X = np.concatenate( (min_X, median_X, max_X, min_T, median_T, max_T), axis=1 )
            
        ## Split data to set train and set test
        self.X_train, self.y_train = dataset_X[0:train_idx], dataset_y[0:train_idx]
        self.X_validation, self.y_validation = dataset_X[train_idx:validation_idx], dataset_y[train_idx:validation_idx]
        self.X_test, self.y_test = dataset_X[validation_idx:], dataset_y[validation_idx:]
        
#        print("Processing data done!!!")
    
    
    def clustering_data(self):
        train_X = copy.deepcopy(self.X_train)
        stimulation_level, positive_number, alpha, beta = self.stimulation_level, self.positive_number, self.alpha, self.beta 
        ### Qua trinh train va dong thoi tao cac hidden unit (Pha 1 - cluster data)
        # 2. Khoi tao hidden thu 1
        temp = randint(0, len(train_X)-1)
        if self.method_statistic == 0:
            hu1 = [0, copy.deepcopy(train_X[temp][:self.sliding]), copy.deepcopy(train_X[temp][self.sliding:])]
        if self.method_statistic == 1:
            hu1 = [0, copy.deepcopy(train_X[temp][:1]), copy.deepcopy(train_X[temp][1:])]
        if self.method_statistic == 2 or self.method_statistic == 3:
            hu1 = [0, copy.deepcopy(train_X[temp][:3]), copy.deepcopy(train_X[temp][3:])]
            
        list_hu = [copy.deepcopy(hu1)]         # list hidden units
        weight_hji = copy.deepcopy(hu1[1]).reshape(1, hu1[1].shape[0])     
        weight_hjk = copy.deepcopy(hu1[2]).reshape(1, hu1[2].shape[0])
        
        self.number_input_cpu = len(hu1[1])
        self.number_input_ram = len(hu1[2])
        #    training_detail_file_name = full_path + 'SL=' + str(stimulation_level) + '_Slid=' + str(sliding) + '_Epoch=' + str(epoch) + '_BS=' + str(batch_size) + '_LR=' + str(learning_rate) + '_PN=' + str(positive_number) + '_CreateHU.txt'
        m = 0
        while m < len(train_X):
            list_dist_mj = []      # Danh sach cac dist(mj)
             # number of hidden units
            for j in range(0, len(list_hu)):                # j: la chi so cua hidden thu j
                dist_sum, dist_1, dist_2 = 0.0, 0.0, 0.0
                for i in range(0, self.number_input_cpu):        # i: la chi so cua input unit thu i
                    dist_1 += pow(train_X[m][i] - weight_hji[j][i], 2.0)
                for i in range(0, self.number_input_ram):
                    dist_2 += pow(train_X[m][self.number_input_cpu + i] - weight_hjk[j][i], 2.0)
                dist_sum = alpha * sqrt(dist_1) + beta * sqrt(dist_2)
                list_dist_mj.append([j, dist_sum])
            list_dist_mj_sorted = sorted(list_dist_mj, key=itemgetter(1))        # Sap xep tu be den lon
            
            c = list_dist_mj_sorted[0][0]      # c: Chi so (index) cua hidden unit thu c ma dat khoang cach min
            distmc = list_dist_mj_sorted[0][1] # distmc: Gia tri khoang cach nho nhat
            if distmc < stimulation_level:
                list_hu[c][0] += 1                  # update hidden unit cth
                hic = exp(- (distmc * distmc) )
                
                neighbourhood_node = 1 + int( 0.10 * len(list_hu) )
                for i in range(0, neighbourhood_node ):
                    c_temp = list_dist_mj_sorted[i][0]
                    delta_1 = (positive_number * hic) * (train_X[m][:self.number_input_cpu] - list_hu[c_temp][1])
                    delta_2 = (positive_number * hic) * (train_X[m][self.number_input_cpu:] - list_hu[c_temp][2])
                    
                    list_hu[c_temp][1] += delta_1
                    list_hu[c_temp][2] += delta_2
                    weight_hji[c_temp] += delta_1
                    weight_hjk[c_temp] += delta_2
                # Tiep tuc vs cac example khac
                m += 1
#                if m % 100 == 0:
#                    print "distmc = {0}".format(distmc)
#                    print "m = {0}".format(m)
            else:
#                print "Failed !!!. distmc = {0}".format(distmc)
                list_hu.append([0, copy.deepcopy(train_X[m][:self.number_input_cpu]), copy.deepcopy(train_X[m][self.number_input_cpu:]) ])
#                print "Hidden unit thu: {0} duoc tao ra.".format(len(list_hu))
                weight_hji = np.append(weight_hji, [copy.deepcopy(train_X[m][:self.number_input_cpu])], axis = 0)
                weight_hjk = np.append(weight_hjk, [copy.deepcopy(train_X[m][self.number_input_cpu:])], axis = 0)
                for hu in list_hu:
                    hu[0] = 0
                # then go to step 1
                m = 0
                if len(list_hu) > 30:
                    break
                ### +++
        ### +++ Get the last matrix weight 
        self.weight_hji, self.weight_hjk, self.list_hu = copy.deepcopy(weight_hji), copy.deepcopy(weight_hjk), copy.deepcopy(list_hu)
        self.len_list_hu = len(list_hu)
#        print("Encoder features done!!!")

    
    def transform_features(self):
        temp1 = []
        for m in range(0, len(self.X_train)):  
            Xhj = []
            for j in range(0, len(self.weight_hji)):     # (w11, w21) (w12, w22), (w13, w23)
                Vhj = Model.distance_func(self.weight_hji[j], self.X_train[m][:self.number_input_cpu])
                Zhj = Model.distance_func(self.weight_hjk[j], self.X_train[m][self.number_input_cpu:])
                Dhj = self.alpha * Vhj + self.beta * Zhj
                Xhj.append(np.tanh(Dhj))
            temp1.append(np.array(Xhj))
            
        temp2 = []
        for m in range(0, len(self.X_test)):  
            Xhj = []
            for j in range(0, len(self.weight_hji)):     # (w11, w21) (w12, w22), (w13, w23)
                Vhj = Model.distance_func(self.weight_hji[j], self.X_test[m][:self.number_input_cpu])
                Zhj = Model.distance_func(self.weight_hjk[j], self.X_test[m][self.number_input_cpu:])
                Dhj = self.alpha * Vhj + self.beta * Zhj
                Xhj.append(np.tanh(Dhj))
            temp2.append(np.array(Xhj))
            
        temp3 = []
        for m in range(0, len(self.X_validation)):  
            Xhj = []
            for j in range(0, len(self.weight_hji)):     # (w11, w21) (w12, w22), (w13, w23)
                Vhj = Model.distance_func(self.weight_hji[j], self.X_validation[m][:self.number_input_cpu])
                Zhj = Model.distance_func(self.weight_hjk[j], self.X_validation[m][self.number_input_cpu:])
                Dhj = self.alpha * Vhj + self.beta * Zhj
                Xhj.append(np.tanh(Dhj))
            temp3.append(np.array(Xhj))
            
        self.S_train = np.array(temp1)
        self.S_test = np.array(temp2)
        self.S_validation = np.array(temp3)
        print("Transform features done!!!")
        
    
    def get_average_mae(self, weightHO, X_data, y_data):
        mae_list = []        
        
        if self.activation == 0:    
            for i in range(0, len(X_data)):
                y_out = np.dot(X_data[i], weightHO)
                mae_list.append( abs( Model.elu_activation(y_out) - y_data[i] ) )
                
        if self.activation == 1:
            for i in range(0, len(X_data)):
                y_out = np.dot(X_data[i], weightHO)
                mae_list.append( abs( Model.relu_activation(y_out) - y_data[i] ) )
                
        if self.activation == 2:
            for i in range(0, len(X_data)):
                y_out = np.dot(X_data[i], weightHO)
                mae_list.append( abs( Model.tanh_activation(y_out) - y_data[i] ) )
                
        if self.activation == 3:
            for i in range(0, len(X_data)):
                y_out = np.dot(X_data[i], weightHO)
                mae_list.append( abs( Model.sigmoid_activation(y_out) - y_data[i] ) )
            
        temp = reduce(add, mae_list, 0)
        return temp[0] / ( len(X_data) * 1.0)
        
    def grade_pop(self, pop):
        """ Find average fitness for a population"""
        summed = reduce(add, (self.fitness_encode(indiv) for indiv in pop) )
        return summed / (len(pop) * 1.0)
    
    def get_global_best(self, pop):
        sorted_pop = sorted(pop, key=lambda temp: temp[3])
        gbest = copy.deepcopy( [sorted_pop[0][0], sorted_pop[0][3] ])
        return gbest
    
    def individual(self, length, min=-1, max=1): 
        """
        x: vi tri hien tai cua con chim
        x_past_best: vi tri trong qua khu ma` ga`n voi thuc an (best result) nhat
        v: vector van toc cua con chim (cung so chieu vs x)
        """
        x = (max - min) * np.random.random_sample((length, 1)) + min 
        x_past_best = copy.deepcopy(x)
        v = np.zeros((x.shape[0], 1))
        x_fitness = self.fitness_individual(x)
        x_past_fitness = copy.deepcopy(x_fitness)
        return [ x, x_past_best, v, x_fitness, x_past_fitness] 
    
    def population(self, pop_size, length, min=-1, max=1):
        """
        individual: 1 solution
        pop_size: number of individuals (population)
        length: number of values per individual
        """
        return [ self.individual(length, min, max) for x in range(pop_size) ]
    
    def fitness_encode(self, encode):
        """ distance between the sum of an indivuduals numbers and the target number. Lower is better"""
        averageTrainMAE = self.get_average_mae(encode[0], self.S_train, self.y_train)
        averageValidationMAE = self.get_average_mae(encode[0], self.S_validation, self.y_validation)
#        sum = reduce(add, individual)  # sum = reduce( (lambda tong, so: tong + so), individual )
        return 0.4*averageTrainMAE + 0.6*averageValidationMAE
    
    def fitness_individual(self, individual):
        """ distance between the sum of an indivuduals numbers and the target number. Lower is better"""
        averageTrainMAE = self.get_average_mae(individual, self.S_train, self.y_train)
        averageValidationMAE = self.get_average_mae(individual, self.S_validation, self.y_validation)
        return (0.4*averageTrainMAE + 0.6*averageValidationMAE)
    
    def build_model_and_train(self):
        """
        - Khoi tao quan the (tinh ca global best)
        - Di chuyen va update vi tri, update gbest
        """
        pop = self.population(self.pop_size, len(self.list_hu), self.value_min, self.value_max)
        fitness_history = []
        
        gbest = self.get_global_best(pop)
        fitness_history.append(gbest[1])
        
        for i in range(self.move_count + 1):
            # Update weight after each move count  (weight down)
            w = (self.move_count - i) / self.move_count * (self.w_max - self.w_min) + self.w_min
            
            for j in range(self.pop_size):
                r1 = np.random.random_sample()
                r2 = np.random.random_sample()
                vi_sau = w * pop[j][2] + self.c1*r1*( pop[j][1] - pop[j][0] ) + self.c2*r2*( gbest[0] - pop[j][0] )
                xi_sau = pop[j][0] + vi_sau     # Xi(sau) = Xi(truoc) + Vi(sau) * deltaT (deltaT = 1)
                fit_sau = self.fitness_individual(xi_sau)
                fit_truoc = pop[j][4]
                # Cap nhat x hien tai, v hien tai, so sanh va cap nhat x past best voi x hien tai
                pop[j][0] = copy.deepcopy(xi_sau)
                pop[j][2] = copy.deepcopy(vi_sau)
                pop[j][3] = fit_sau
        
                if fit_sau < fit_truoc:
                    pop[j][1] = copy.deepcopy(xi_sau)
                    pop[j][4] = fit_sau
            
            gbest = self.get_global_best(pop)
            fitness_history.append(gbest[1])
#            print "Generation : {0}, average MAE over population: {1}".format(i+1, gbest[1])
        
        self.weight, self.loss_train = gbest[0], fitness_history
#        print("Build model and train done!!!")
        
        
    def predict(self):
        # Evaluate models on the test set
        X_test, y_test = self.S_test, self.y_test
        
        X_size = X_test.shape[1]   
        y_size = y_test.shape[1]
        
        X = tf.placeholder("float64", shape=[None, X_size], name='X')
        y = tf.placeholder("float64", shape=[None, y_size], name='y')
        
        W = tf.Variable(self.weight)
        if self.activation == 0:    
            y_ = tf.nn.elu( tf.matmul(X, W) )
        if self.activation == 1:
            y_ = tf.nn.relu( tf.matmul(X, W) )
        if self.activation == 2:
            y_ = tf.nn.tanh( tf.matmul(X, W) )
        if self.activation == 3:
            y_ = tf.nn.sigmoid( tf.matmul(X, W) )
        
        # Calculate the predicted outputs
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            y_est_np = sess.run(y_, feed_dict={X: X_test, y: y_test})
            
            y_test_inverse = self.min_max_scaler.inverse_transform(y_test)
            y_pred_inverse = self.min_max_scaler.inverse_transform(y_est_np)
            
            testScoreRMSE = sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
            testScoreMAE = mean_absolute_error(y_test_inverse, y_pred_inverse)
            
            self.y_predict, self.score_test_RMSE, self.score_test_MAE = y_est_np, testScoreRMSE, testScoreMAE
            self.y_test_inverse, self.y_pred_inverse = y_test_inverse, y_pred_inverse
            
#            print('DONE - RMSE: %.5f, MAE: %.5f' % (testScoreRMSE, testScoreMAE))
        
#        print("Predict done!!!")

    def draw_loss(self):
        plt.figure(1)
        plt.plot(range(len(self.loss_train)), self.loss_train, label="Loss on training per epoch")
        plt.xlabel('Iteration', fontsize=12)  
        plt.ylabel('Loss', fontsize=12)          
        
    def draw_predict(self):
        plt.figure(2)
        plt.plot(self.y_test_inverse)
        plt.plot(self.y_pred_inverse)
        plt.title('Model predict')
        plt.ylabel('Real value')
        plt.xlabel('Point')
        plt.legend(['realY... Test Score RMSE= ' + str(self.score_test_RMSE) , 'predictY... Test Score MAE= '+ str(self.score_test_MAE)], loc='upper right')
        plt.savefig(pathsave + self.filenamesave + ".png")
        plt.close()
        
    def save_file_csv(self):
        t1 = np.concatenate( (self.y_test_inverse, self.y_pred_inverse), axis = 1)
        np.savetxt(pathsave + self.filenamesave + ".csv", t1, delimiter=",")

    
    def fit(self):
        self.preprocessing_data()
        self.clustering_data()
        if self.len_list_hu <= 30:
            self.transform_features()
            self.build_model_and_train()
            self.predict()
            if self.score_test_MAE < 0.30:
        #        self.draw_loss()
                self.draw_predict()
                self.save_file_csv()
    
    @staticmethod
    def distance_func(a, b):
        return distance.euclidean(a, b)
    
    @staticmethod
    def sigmoid_activation(x):
        return 1.0 / (1.0 + exp(-x))
    
    @staticmethod
    def elu_activation(x):
        return (exp(x) - 1.0) if x < 0 else x 
        
    @staticmethod
    def relu_activation(x):
        return max(x, 0)
    
    @staticmethod 
    def tanh_activation(x):
        return np.tanh(x)
    
    @staticmethod
    def get_random_input_vector(train_X):
        return copy.deepcopy(train_X[randint(0, len(train_X)-1)])
    
    @staticmethod
    def get_batch_data_next(trainX, trainY, index, batch_size):
        real_index = index*batch_size
        if (len(trainX) % batch_size != 0 and index == (len(trainX)/batch_size +1) ):
            return (trainX[real_index:], trainY[real_index:])
        elif (real_index == len(trainX)):
            return ([], [])
        else:
            return (trainX[real_index: (real_index+batch_size)], trainY[real_index: (real_index+batch_size)])

    @staticmethod
    def get_mutate_vector_weight(wHa, wHb, mutation_id = 1):
        temp = []
        if mutation_id == 1:    # Lay trung binh cong
            for i in range(len(wHa)):
                temp.append( (wHa[i] + wHb[i]) / 2 )
        if mutation_id == 2:    # Lay uniform
            for i in range(len(wHa)):
                temp.append(uniform(wHa[i], wHb[i]))
        return np.array(temp)
    

pathsave = "/home/hunter/nguyenthieu95/ai/6_google_trace/SONIA/testing_new/cluster_pso/no_mutation/2m/result/cpu_ram_2para/"
fullpath = "/home/hunter/nguyenthieu95/ai/data/GoogleTrace/"
filename = "data_resource_usage_twoMinutes_6176858948.csv"

#pathsave = "/home/thieunv/university/LabThayMinh/code/6_google_trace/SONIA/testing_new/cluster_pso/no_mutation/2m/result/cpu_ram_2para/"
#fullpath = "/home/thieunv/university/LabThayMinh/code/data/GoogleTrace/"
#filename = "data_resource_usage_twoMinutes_6176858948.csv"

df = read_csv(fullpath+ filename, header=None, index_col=False, usecols=[3, 5], engine='python')   
dataset_original = df.values

stimulation_levels = [0.15, 0.25, 0.35, 0.45, 0.55, 0.7]  #[0.10, 0.2, 0.25, 0.50, 1.0, 1.5, 2.0]  # [0.20]    
positive_numbers = [0.1, 0.2, 0.35] #[0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.20]     # [0.1]
distance_levels = [0.25, 0.4, 0.55]
sliding_windows = [2, 3, 5] #[ 2, 3, 5]           # [3]  
list_num = [(12000, 20000)]
method_statistics = [0]
activations = [0, 1, 2, 3]  # elu, relu, tanh, sigmoid
d_couple = [(0.7, 0.3), (0.2, 0.8)]

w_min = 0.4       # [0-1] -> [0.4-0.9]      Trong luong cua con chim
w_max = 0.9
c_couple = [(1.2, 1.2), (2, 2), (0.8, 2.0), (1.6, 0.6)]         #c1, c2 = 2, 2       # [0-2]   Muc do anh huong cua local va global 
# r1, r2 : random theo tung vong lap
# delta(t) = 1 (do do: x(sau) = x(truoc) + van_toc
pop_sizes = [100, 200, 300, 400]      # Kich thuoc quan the
move_counts = [50, 100, 200, 300]     # So lan di chuyen` 
value_min = -1      # value min of weight 
value_max = +1



pl1 = 1         # Use to draw figure
#pl2 = 1000
so_vong_lap = 0

for list_idx in list_num:
    for sliding in sliding_windows:
        for method_statistic in method_statistics:
            for acti in activations:
                for sti_level in stimulation_levels:
                    for dis_level in distance_levels:
                        for positive_number in positive_numbers:
                            
                            for pop_size in pop_sizes:
                                for move_count in move_counts:
                                    for c_id in c_couple:
                                        
                                        for d_id in d_couple:

                                            pso_param = [pop_size, move_count, value_min, value_max, w_min, w_max, c_id[0], c_id[1]]
                                
                                            febpnn = Model(dataset_original, list_idx, pso_param, sliding, positive_number, sti_level, dis_level, method_statistic, acti, d_id)
                                            febpnn.fit()
                                            
                                            so_vong_lap += 1
                                            if so_vong_lap % 1000 == 0:
                                                print "Vong lap thu : {0}".format(so_vong_lap)

print "Processing DONE !!!"
    
    