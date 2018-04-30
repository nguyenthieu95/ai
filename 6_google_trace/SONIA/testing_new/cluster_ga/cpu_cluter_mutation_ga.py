#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 11:08:48 2018

@author: thieunv

- Chay cung` tham so: mutation cho ket qua tot hon duoc 1-2% 

Mutation: 0.23
No-mutation: 0.247

- Cluster + Mutation + GA 


*) GradientDescentOptimizer (MinMaxScaler - 2)

    elu / relu/ tanh/ sigmoid ===> failed/ failed/ failed / (0.48) failed 
    
*) AdamOptimizer 
    
     elu / relu/ tanh/ sigmoid ===> 0.42/ failed / 0.42 / failed
    
*) AdagradOptimizer   
    
     elu / relu/  tanh/  sigmoid ===> 0.45/ 0.45 / 0.46 (g00d) / failed

*) AdadeltaOptimizer
    
     elu / relu/ tanh/ sigmoid ===> 0.41/ 0.41 / 0.41 / 0.48
    
=====> No Tech, Gauss
"""

import tensorflow as tf
import numpy as np
from scipy.spatial import distance
from math import exp, sqrt, ceil
import copy
from random import randint, uniform, random
from operator import itemgetter, add
from pandas import read_csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Model(object):
    def __init__(self, dataset_original, list_idx, ga_param, sliding, positive_number, sti_level, dis_level = 0.25, method_statistic = 0):
        self.dataset_original = dataset_original
        self.train_idx = list_idx[0]
        self.test_idx = list_idx[1]
        self.pop_size = ga_param[0]
        self.gen_count = ga_param[1]
        self.value_min = ga_param[2]
        self.value_max = ga_param[3]
        self.retain = ga_param[4]
        self.random_select = ga_param[5]
        self.mutate = ga_param[6]
        self.sliding = sliding
        self.positive_number = positive_number
        self.stimulation_level = sti_level
        self.distance_level = dis_level
        self.method_statistic = method_statistic
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.standard_scaler = preprocessing.StandardScaler()
        
    def preprocessing_data(self):
        train_idx, test_idx, dataset_original, sliding = self.train_idx, self.test_idx, self.dataset_original, self.sliding
        ## Transform all dataset
        dataset_split = dataset_original[:test_idx + sliding]               # Save orginal data to take y_test
        dataset_transform = self.min_max_scaler.fit_transform(dataset_split)
        
        ## Handle data with sliding
        dataset_sliding = dataset_transform[:len(dataset_transform)-sliding]
        for i in range(sliding-1):
            dddd = np.array(dataset_transform[i+1: len(dataset_transform)-sliding+i+1])
            dataset_sliding = np.concatenate((dataset_sliding, dddd), axis=1)
            
        ## window value: x1 \ x2 \ x3  (dataset_sliding)
        ## Now we using different method on this window value 
        dataset_y = copy.deepcopy(dataset_transform[sliding:])      # Now we need to find dataset_X
        
        if self.method_statistic == 0:
            dataset_X = copy.deepcopy(dataset_sliding)

        if self.method_statistic == 1:
            """
            mean(x1, x2, x3, ...)
            """
            dataset_X = np.reshape(np.mean(dataset_sliding, axis = 1), (-1, 1))
            
        if self.method_statistic == 2:
            """
            min(x1, x2, x3, ...), mean(x1, x2, x3, ...), max(x1, x2, x3, ....)
            """
            min_X = np.reshape(np.amin(dataset_sliding, axis = 1), (-1, 1))
            mean_X = np.reshape(np.mean(dataset_sliding, axis = 1), (-1, 1))
            max_X = np.reshape(np.amax(dataset_sliding, axis = 1), (-1, 1))
            dataset_X = np.concatenate( (min_X, mean_X, max_X), axis=1 )
            
        if self.method_statistic == 3:
            """
            min(x1, x2, x3, ...), median(x1, x2, x3, ...), max(x1, x2, x3, ....)
            """
            min_X = np.reshape(np.amin(dataset_sliding, axis = 1), (-1, 1))
            median_X = np.reshape(np.median(dataset_sliding, axis = 1), (-1, 1))
            max_X = np.reshape(np.amax(dataset_sliding, axis = 1), (-1, 1))
            dataset_X = np.concatenate( (min_X, median_X, max_X), axis=1 )
            
        ## Split data to set train and set test
        self.X_train, self.y_train = dataset_X[0:train_idx], dataset_y[0:train_idx]
        self.X_test, self.y_test = dataset_X[train_idx:], dataset_y[train_idx:]
        
        print("Processing data done!!!")
    
    
    def clustering_data(self):
        train_X = copy.deepcopy(self.X_train)
        stimulation_level, positive_number = self.stimulation_level, self.positive_number
        
        ### Qua trinh train va dong thoi tao cac hidden unit (Pha 1 - cluster data)
        # 2. Khoi tao hidden thu 1
        hu1 = [0, Model.get_random_input_vector(train_X)]   # hidden unit 1 (t1, wH)
        list_hu = [copy.deepcopy(hu1)]         # list hidden units
        matrix_Wih = copy.deepcopy(hu1[1]).reshape(1, hu1[1].shape[0])     # Mang 2 chieu 
        #    training_detail_file_name = full_path + 'SL=' + str(stimulation_level) + '_Slid=' + str(sliding) + '_Epoch=' + str(epoch) + '_BS=' + str(batch_size) + '_LR=' + str(learning_rate) + '_PN=' + str(positive_number) + '_CreateHU.txt'
        m = 0
        while m < len(train_X):
            list_dist_mj = []      # Danh sach cac dist(mj)
             # number of hidden units
            for j in range(0, len(list_hu)):                # j: la chi so cua hidden thu j
                dist_sum = 0.0
                for i in range(0, len(train_X[0])):        # i: la chi so cua input unit thu i
                    dist_sum += pow(train_X[m][i] - matrix_Wih[j][i], 2.0)
                list_dist_mj.append([j, sqrt(dist_sum)])
            list_dist_mj = sorted(list_dist_mj, key=itemgetter(1))        # Sap xep tu be den lon
            
            c = list_dist_mj[0][0]      # c: Chi so (index) cua hidden unit thu c ma dat khoang cach min
            distmc = list_dist_mj[0][1] # distmc: Gia tri khoang cach nho nhat
            
            if distmc < stimulation_level:
                list_hu[c][0] += 1                  # update hidden unit cth
                
                # Find Neighbourhood
                list_distjc = []
                for i in range(0, len(matrix_Wih)):
                    list_distjc.append([i, Model.distance_func(matrix_Wih[c], matrix_Wih[i])])
                list_distjc = sorted(list_distjc, key=itemgetter(1))
                
                # Update BMU (Best matching unit and it's neighbourhood)
                neighbourhood_node = int( 1 + ceil( 0.2 * (len(list_hu) - 1) ) )
                for i in range(0, neighbourhood_node ):
                    if i == 0:
                        list_hu[c][1] += (positive_number * distmc) * (train_X[m] - list_hu[c][1])
                        matrix_Wih[c] += (positive_number * distmc) * (train_X[m] - list_hu[c][1])
                    else:
                        c_temp = list_distjc[i][0]
                        distjc = list_distjc[i][1]
                        hic = exp(-distjc * distjc)
                        delta = (positive_number * hic) * (train_X[m] - list_hu[c_temp][1])
                        
                        list_hu[c_temp][1] += delta
                        matrix_Wih[c_temp] += delta
                
                # Tiep tuc vs cac example khac
                m += 1
                if m % 1000 == 0:
                    print "distmc = {0}".format(distmc)
                    print "m = {0}".format(m)
            else:
                print "Failed !!!. distmc = {0}".format(distmc)
                list_hu.append([0, copy.deepcopy(train_X[m]) ])
                print "Hidden unit thu: {0} duoc tao ra.".format(len(list_hu))
                matrix_Wih = np.append(matrix_Wih, [copy.deepcopy(train_X[m])], axis = 0)
                for hu in list_hu:
                    hu[0] = 0
                # then go to step 1
                m = 0
                ### +++
        ### +++ Get the last matrix weight 
        self.matrix_Wih = copy.deepcopy(matrix_Wih)
        self.list_hu = copy.deepcopy(list_hu)

        print("Encoder features done!!!")
        
    def mutation_hidden_node(self):
        self.threshold_number = int (len(self.X_train) / len(self.list_hu))
        ### Qua trinh mutated hidden unit (Pha 2- Adding artificial local data)
        # Adding 2 hidden unit in begining and ending points of input space
        t1 = np.zeros(self.sliding)
        t2 = np.ones(self.sliding)
        self.list_hu.append([0, t1])
        self.list_hu.append([0, t2])
        
        self.matrix_Wih = np.concatenate((self.matrix_Wih, np.array([t1])), axis=0)
        self.matrix_Wih = np.concatenate((self.matrix_Wih, np.array([t2])), axis=0)
    
    #    # Sort matrix weights input and hidden, Sort list hidden unit by list weights
        for i in range(0, self.matrix_Wih.shape[1]):
            self.matrix_Wih = sorted(self.matrix_Wih, key=lambda elem_list: elem_list[i])
            self.list_hu = sorted(self.list_hu, key=lambda elem_list: elem_list[1][i])
             
            for i in range(len(self.list_hu) - 1):
                ta, wHa = self.list_hu[i][0], self.list_hu[i][1]
                tb, wHb = self.list_hu[i+1][0], self.list_hu[i+1][1]
                dab = Model.distance_func(wHa, wHb)
                
                if dab > self.distance_level and ta < self.threshold_number and tb < self.threshold_number:
                    # Create new mutated hidden unit (Dot Bien)
                    temp_node = Model.get_mutate_vector_weight(wHa, wHb, mutation_id=1)
                    self.list_hu.insert(i+1, [0, copy.deepcopy(temp_node)])
                    self.matrix_Wih = np.insert(self.matrix_Wih, [i+1], temp_node, axis=0)
                    print "New hidden unit created. {0}".format(len(self.matrix_Wih))
        print("Finished mutation hidden unit!!!")
    
    def transform_features(self):
        temp1 = []
        for i in range(0, len(self.X_train)):  
            Sih = []
            for j in range(0, len(self.matrix_Wih)):     # (w11, w21) (w12, w22), (w13, w23)
                Sih.append(np.tanh( Model.distance_func(self.matrix_Wih[j], self.X_train[i])))
            temp1.append(np.array(Sih))
        
        temp2 = []
        for i in range(0, len(self.X_test)):  
            Sih = []
            for j in range(0, len(self.matrix_Wih)):     # (w11, w21) (w12, w22), (w13, w23)
                Sih.append(np.tanh( Model.distance_func(self.matrix_Wih[j], self.X_test[i])))
            temp2.append(np.array(Sih))
            
        self.S_train = np.array(temp1)
        self.S_test = np.array(temp2)
        
        print("Transform features done!!!")
    
    def get_average_mae(self, weightHO):
        mae_list = []        
        for i in range(0, len(self.S_train)):
            y_out = np.dot(self.S_train[i], weightHO)
            mae_list.append( abs( y_out - self.y_train[i] ) )
        temp = reduce(add, mae_list, 0)
        return temp[0] / ( len(self.S_train) * 1.0)
        
    def individual(self, length, min=-1, max=1): 
        return (max - min) * np.random.random_sample((length, 1)) + min 
    
    def population(self, pop_size, length, min=-1, max=1):
        """
        individual: 1 solution
        pop_size: number of individuals (population)
        length: number of values per individual
        """
        return [ self.individual(length, min, max) for x in range(pop_size) ]
    
    def fitness(self, individual):
        """ distance between the sum of an indivuduals numbers and the target number. Lower is better"""
        averageMAE = self.get_average_mae(individual)
#        sum = reduce(add, individual)  # sum = reduce( (lambda tong, so: tong + so), individual )
        return averageMAE
    
    def grade(self, pop):
        """ Find average fitness for a population"""
        summed = reduce(add, (self.fitness(indiv) for indiv in pop) )
        return summed / (len(pop) * 1.0)
    
    def evolve(self, pop, retain=0.2, random_select=0.05, mutate=0.01):
        graded = [ (self.fitness(indiv), indiv) for indiv in pop ]
        graded = sorted(graded, key=lambda temp : temp[0])
        graded = [ x[1] for x in graded ]
        retain_length = int(len(graded) * retain)
        parents = graded[:retain_length]
        
        # Randomly add other individuals to promote genetic diversity
        for individual in graded[retain_length:]:
            if random_select > random():
                parents.append(individual)
        
        # Mutate some individuals
        for individual in parents:  # Dot bien cha me (1/100)
            if mutate > random():
                pos_to_mutate = randint(0, len(individual) - 1)
                minIndiv = min(individual)
                maxIndiv = max(individual)
                individual[pos_to_mutate] = (maxIndiv - minIndiv) * np.random.random_sample(1) + minIndiv
            
        # Crossover parents to create children
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []
        while len(children) < desired_length:
            male = randint(0, parents_length - 1)   # father
            female = randint(0, parents_length - 1) # mother
            if male != female:
                male = parents[male]
                female = parents[female]
                half = len(male) / 2
                child = np.concatenate( (male[:half], female[half:]), axis=0 )
                children.append(child)
        
        parents.extend(children)
        return parents
    
    
    def build_model_and_train(self):
        pop = self.population(self.pop_size, len(self.list_hu), self.value_min, self.value_max)
        fitness_history = [self.grade(pop)]
        for i in range(self.gen_count):
            pop = self.evolve(pop, self.retain, self.random_select, self.mutate)
            temp = self.grade(pop)
            fitness_history.append(temp)
            print "Generation : {0}, average MAE over population: {1}".format(i+1, temp)
        
        graded = [ (self.fitness(indiv), indiv) for indiv in pop ]
        graded = sorted(graded, key=lambda temp : temp[0])
        self.weight, self.loss_train = graded[0][1], fitness_history
        print("Build model and train done!!!")
        
        
    def predict(self):
        # Evaluate models on the test set
        X_test, y_test = self.S_test, self.y_test
        
        X_size = X_test.shape[1]   
        y_size = y_test.shape[1]
        
        X = tf.placeholder("float64", shape=[None, X_size], name='X')
        y = tf.placeholder("float64", shape=[None, y_size], name='y')
        
        W = tf.Variable(self.weight)
        y_ = tf.nn.elu( tf.matmul(X, W) )
        
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
            
            print('DONE - RMSE: %.5f, MAE: %.5f' % (testScoreRMSE, testScoreMAE))
        
        print("Predict done!!!")

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
    
    def fit(self):
        self.preprocessing_data()
        self.clustering_data()
        self.mutation_hidden_node()
        self.transform_features()
        self.build_model_and_train()
        self.predict()
        self.draw_loss()
        self.draw_predict()
    
    @staticmethod
    def distance_func(a, b):
        return distance.euclidean(a, b)
    
    @staticmethod
    def sigmoid_activation(x):
        return 1.0 / (1.0 + exp(-x))
    
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
    

## Load data frame
#full_path_name="/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/data/"
#full_path= "/mnt/volume/ggcluster/spark-2.1.1-bin-hadoop2.7/thieunv/machine_learning/6_google_trace/FLNN/results/notDecompose/data10minutes/univariate/cpu/"
#file_name = "Fuzzy_data_sampling_617685_metric_10min_datetime_origin.csv"



fullpath = "/home/thieunv/university/LabThayMinh/code/data/GoogleTrace/"
filename = "data_resource_usage_twoMinutes_6176858948.csv"
df = read_csv(fullpath+ filename, header=None, index_col=False, usecols=[3], engine='python')   
dataset_original = df.values

stimulation_levels = [0.35]  #[0.10, 0.2, 0.25, 0.50, 1.0, 1.5, 2.0]  # [0.20]    
positive_numbers = [0.15] #[0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.20]     # [0.1]
distance_levels = [0.35]
sliding_windows = [2] #[ 2, 3, 5]           # [3]  
list_num = [(15000, 20000)]

pop_size = 150      # Kich thuoc quan the
gen_count = 120     # So vong doi` 
value_min = -1      # value min of weight 
value_max = +1
retain=0.2          # % Giu lai trong qua trinh loc bo quan the
random_select=0.07  # % Lay tu phan bi loai bo, muc dich de lam tang tinh da dang cua quan the
mutate=0.02         # % Dot bien trong parents.
ga_param = [pop_size, gen_count, value_min, value_max, retain, random_select, mutate]


#fullpath = "/home/thieunv/university/LabThayMinh/code/data/preprocessing/"
#filename = "time_series_cpu.csv"
#df = read_csv(fullpath+ filename, header=0, index_col=False, usecols=[1], engine='python', dtype=np.float64)   
#dataset_original = df.values
#stimulation_level = [0.8]  #[0.10, 0.2, 0.25, 0.50, 1.0, 1.5, 2.0]  # [0.20]    
#positive_numbers = [0.15] #[0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.20]     # [0.1]   
#learning_rates = [0.25] #[0.005, 0.01, 0.025, 0.05, 0.10, 0.12, 0.15]   # [0.2]    
#sliding_windows = [5] #[ 2, 3, 5]           # [3]  
#epochs = [1300] #[100, 250, 500, 1000, 1500, 2000]   # [500]                       
#batch_sizes = [32] #[8, 16, 32, 64, 128]      # [16]     
#list_num = [(3200, 4100)]




#fullpath = "/home/thieunv/university/LabThayMinh/code/data/preprocessing/"
#filename = "daily-minimum-temperatures-in-me.csv"
#df = read_csv(fullpath+ filename, header=0, index_col=False, usecols=[1], engine='python')   
#dataset_original = df.values
#stimulation_level = [0.35]  #[0.10, 0.2, 0.25, 0.50, 1.0, 1.5, 2.0]  # [0.20]    
#positive_numbers = [0.15] #[0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.20]     # [0.1]   
#learning_rates = [0.35] #[0.005, 0.01, 0.025, 0.05, 0.10, 0.12, 0.15]   # [0.2]    
#sliding_windows = [3] #[ 2, 3, 5]           # [3]  
#epochs = [2800] #[100, 250, 500, 1000, 1500, 2000]   # [500]                       
#batch_sizes = [32] #[8, 16, 32, 64, 128]      # [16]     
#list_num = [(2700, 3600)]




#time_stamp','taskIndex','machineId','meanCPUUsage','CMU','AssignMem',
#'unmapped_cache_usage','page_cache_usage', 'max_mem_usage','mean_diskIO_time',
#'mean_local_disk_space','max_cpu_usage', 'max_disk_io_time', 'cpi', 'mai',
#'sampling_portion','agg_type','sampled_cpu_usage
# 4-cpu(god) 5- ram(good), 6- disk io(god), 7-disk time(god)

pl1 = 1         # Use to draw figure
#pl2 = 1000
so_vong_lap = 0

for list_idx in list_num:
    for sliding in sliding_windows:
        for sti_level in stimulation_levels:
            for dis_level in distance_levels:
                for positive_number in positive_numbers:
                    
                    febpnn = Model(dataset_original, list_idx, ga_param, sliding, positive_number, sti_level, dis_level)
                    febpnn.fit()
                    
                    so_vong_lap += 1
                    if so_vong_lap % 5000 == 0:
                        print "Vong lap thu : {0}".format(so_vong_lap)

print "Processing DONE !!!"
    
    