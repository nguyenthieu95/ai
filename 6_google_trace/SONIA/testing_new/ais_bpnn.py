#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 09:18:54 2018

@author: thieunv

- AIS based BPNN

- Multivariate - Cluster + Mutation + PSO 


- Mutation operator: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.22.4413&rep=rep1&type=pdf

- AIS theory: ftp://ftp.dca.fee.unicamp.br/pub/docs/vonzuben/tr_dca/trdca0199.pdf

"""

import numpy as np
from scipy.spatial import distance
from math import exp, sqrt
import copy
from random import randint, uniform 
from operator import add
from pandas import read_csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Model(object):
    def __init__(self, dataset_original, list_idx, hidden_node, ais_param, sliding, method_statistic = 0):                   
        self.dataset_original = dataset_original
        self.train_idx = list_idx[0]
        self.validation_idx = int(list_idx[0] + (list_idx[1] - list_idx[0])/2)
        self.test_idx = list_idx[1]
        self.p_rt = ais_param[0]
        self.rs = ais_param[1]
        self.g_max = ais_param[2]
        self.low_up_w = ais_param[3]
        self.low_up_b = ais_param[4]
        self.number_node_hidden = hidden_node
        self.sliding = sliding
        self.method_statistic = method_statistic
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.standard_scaler = preprocessing.StandardScaler()
        self.dimension = dataset_original.shape[1]
#        self.filenamesave = "point-slid_{0}-method_{1}-pos_{2}-sti_{3}-dis_{4}-pop_size_{5}-move_count_{6}-c1_{7}-c2_{8}-activ_{9}".format(sliding, method_statistic, positive_number, sti_level, dis_level, pso_param[0], pso_param[1], pso_param[6], pso_param[7], activation)
        
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
            dataset_X = np.reshape(np.mean(dataset_sliding[:, 0:sliding], axis = 1), (-1, 1))
            
        if self.method_statistic == 2:
            """
            min(x1, x2, x3, ...), mean(x1, x2, x3, ...), max(x1, x2, x3, ....)
            """
            min_X = np.reshape(np.amin(dataset_sliding[:, 0:sliding], axis = 1), (-1, 1))
            mean_X = np.reshape(np.mean(dataset_sliding[:, 0:sliding], axis = 1), (-1, 1))
            max_X = np.reshape(np.amax(dataset_sliding[:, 0:sliding], axis = 1), (-1, 1))
            
            dataset_X = np.concatenate( (min_X, mean_X, max_X), axis=1 )
            
        if self.method_statistic == 3:
            """
            min(x1, x2, x3, ...), median(x1, x2, x3, ...), max(x1, x2, x3, ....)
            """
            min_X = np.reshape(np.amin(dataset_sliding[:, 0:sliding], axis = 1), (-1, 1))
            median_X = np.reshape(np.median(dataset_sliding[:, 0:sliding], axis = 1), (-1, 1))
            max_X = np.reshape(np.amax(dataset_sliding[:, 0:sliding], axis = 1), (-1, 1))
            
            dataset_X = np.concatenate( (min_X, median_X, max_X), axis=1 )
            
        ## Split data to set train and set test
        self.X_train, self.y_train = dataset_X[0:train_idx], dataset_y[0:train_idx]
        self.X_validation, self.y_validation = dataset_X[train_idx:validation_idx], dataset_y[train_idx:validation_idx]
        self.X_test, self.y_test = dataset_X[validation_idx:], dataset_y[validation_idx:]
        
        print("Processing data done!!!")
    
   
    
    def create_candidate(self, low_up_weights = [-0.2, 0.8], low_up_biases = [-0.5, 0.5]):
        """
        """
        self.number_node_input = self.X_train.shape[1]
        self.number_node_output = self.y_train.shape[1]
        
        self.size_w1 = self.number_node_input * self.number_node_hidden
        self.size_b1 = self.number_node_hidden
        self.size_w2 = self.number_node_hidden * self.number_node_output
        self.size_b2 = self.number_node_output
        
        W1 = (low_up_weights[1] - low_up_weights[0]) * np.random.random_sample((self.size_w1, 1)) + low_up_weights[0] 
        B1 = (low_up_biases[1] - low_up_biases[0]) * np.random.random_sample((self.size_b1, 1)) + low_up_biases[0] 
        W2 = (low_up_weights[1] - low_up_weights[0]) * np.random.random_sample((self.size_w2, 1)) + low_up_weights[0] 
        B2 = (low_up_biases[1] - low_up_biases[0]) * np.random.random_sample((self.size_b2, 1)) + low_up_biases[0]
        candidate = np.concatenate( (W1, B1, W2, B2), axis=0 )
        print("Create candidate done!!!")
        
        return candidate
    
    def repertoires(self, rs_size = 6):
        """ population
        rs_size: number of repertoire in antibody (population)
        """
        return [ self.create_candidate() for x in range(rs_size) ]
    
    def init_params(self):
        repertoires = self.repertoires()
        return repertoires
    
    def get_mae(self, reperoire, X_data=None, y_data=None):
        mae_list = []        
        self.t1 = self.size_w1 + self.size_b1
        self.t2 = self.size_w1 + self.size_b1 + self.size_w2
        
        w1 = np.reshape(reperoire[:self.size_w1], (self.number_node_input, -1))
        b1 = np.reshape(reperoire[self.size_w1:self.t1], (-1, self.size_b1))
        w2 = np.reshape(reperoire[self.t1:self.t2], (self.number_node_hidden, -1))
        b2 = np.reshape(reperoire[self.t2:], (-1, self.size_b2))  
        
        for i in range(0, len(X_data)):
            vector_hidden = np.matmul(X_data[i], w1)
            vector_hidden = np.add(vector_hidden, b1)
            input_hidden = self.bipolar_sigmoid_activation(vector_hidden)
            output_hidden = np.matmul(input_hidden, w2)
            out = np.add(output_hidden, b2)
            mae_list.append( pow( (y_data[i] - Model.elu_activation(out)), 2 ) )
        temp = reduce(add, mae_list, 0)
        return temp[0] / ( len(X_data) * 1.0)
    
    
    def get_affinity(self, reperoire):
        return -1 * sqrt(self.get_mae(reperoire, self.X_train, self.y_train))
    
    
    def get_ab_with_affinity_max(self, repertoires):
        repers = [ (reper, self.get_affinity(reper)) for reper in repertoires ]
        crep = max(repers, key=lambda x: x[1])
        return crep[0]    
    
    def get_prj(self, repe_max, current_repe):
        list_p = []
        for i in range(len(repe_max)):
            t1 = abs( (repe_max[i] - current_repe[i]) / repe_max[i] )
            list_p.append(1.0 / exp(t1))
        temp = reduce(add, list_p, 0)
        return temp / len(repe_max)
    
    
    def somatic_hypermutation(self, reperoire, current_generation):
        """
        2 task: uniform search and local fine-tuning
        """
        for i in range(len(reperoire)):
            Ag = pow( (np.random.random_sample() * (1.0 - current_generation / self.g_max)), 2 )
            
            if (i < self.size_w1) or (self.t1 <= i and i < self.t2):
                low_up = self.low_up_w
            else:
                low_up = self.low_up_b

            if np.random.random_sample() < 0.5:
                reperoire[i] = reperoire[i] + Ag * (low_up[1] - reperoire[i])
            else:
                reperoire[i] = reperoire[i] - Ag * (reperoire[i] - low_up[0])
    
    def receptor_editing(self, reperoite):
        reperoite = reperoite + pow( (np.random.random_sample()), 3 ) * np.random.standard_cauchy(len(reperoite))
        
    
    def introduce_diverse_abs(self, list_abj):
        male = randint(0, len(list_abj) - 1)   # father
        female = randint(0, len(list_abj) - 1) # mother
        if male != female:
            father = list_abj[male]
            mother = list_abj[female]
            recombining_index = randint(0, len(father) - 1) 
            father[recombining_index], mother[recombining_index] = mother[recombining_index] + np.random.random_sample(), father[recombining_index] + np.random.random_sample()
            list_abj[male] = father
            list_abj[female] = mother
    
    def update_ab_repertoire(self, reperoires, list_promoted_ab):
        new_repers = reperoires + list_promoted_ab
        repers = [ (reper, self.get_affinity(reper)) for reper in new_repers ]
        repers = sorted(repers, key=lambda x: x[1])
        sorted_repers = [ x[0] for x in repers ]
        return sorted_repers[(len(new_repers)-self.rs):]
    
    def evaluate_test_accuracy(self, reper):
        self.get_mae(reper, self.X_test, self.y_test)
    
    def ais(self):
        # Step 1: Initialize parameters
        ### Training stage
        g = 0
        repertoires = self.init_params()    
        
        while g < self.g_max :
            ### Step 2: 
            ab_max = self.get_ab_with_affinity_max(repertoires)
            print("Epoch {0}, RMSE = {1}".format(g+1, self.get_mae(ab_max, self.X_train, self.y_train)))
            
            ### Step 3: Perform clonal selection
            list_promoted_ab = []
            for repe in repertoires:
                if self.get_prj(ab_max, repe) >= self.p_rt:
                    list_promoted_ab.append(copy.deepcopy(repe))
#                else:
#                    self.suppress()
            
            ### Step 4: Implement affinity maturation
            for pro_ab in list_promoted_ab:
                if np.random.random_sample() <= 0.5:
                    self.somatic_hypermutation(pro_ab, g)
                else:
                    self.receptor_editing(pro_ab)
            
            ### Step 5:
            self.introduce_diverse_abs(list_promoted_ab)
            
            ### Step 6: 
            self.update_ab_repertoire(repertoires, list_promoted_ab)
            
            g += 1
        
        #### Test Stage 
        ### Step 7:
        self.predict(repertoires[-1])
        
             
    def fit(self):
        self.preprocessing_data()
        self.ais()
        self.draw_predict()
        
    def predict(self, reperoire):
        w1 = np.reshape(reperoire[:self.size_w1], (self.number_node_input, -1))
        b1 = np.reshape(reperoire[self.size_w1:self.t1], (-1, self.size_b1))
        w2 = np.reshape(reperoire[self.t1:self.t2], (self.number_node_hidden, -1))
        b2 = np.reshape(reperoire[self.t2:], (-1, self.size_b2))  
        
        y_est_np = []
        for i in range(0, len(self.X_test)):
            vector_hidden = np.matmul(self.X_test[i], w1)
            vector_hidden = np.add(vector_hidden, b1)
            input_hidden = self.bipolar_sigmoid_activation(vector_hidden)
            output_hidden = np.matmul(input_hidden, w2)
            out = np.add(output_hidden, b2)
            y_est_np.append(Model.elu_activation(out))
        
        # Evaluate models on the test set
        y_test_inverse = self.min_max_scaler.inverse_transform(self.y_test)
        y_pred_inverse = self.min_max_scaler.inverse_transform(np.array(y_est_np).reshape(-1, 1))
        
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
#        plt.savefig(pathsave + self.filenamesave + ".png")
#        plt.close()
        
    def save_file_csv(self):
        t1 = np.concatenate( (self.y_test_inverse, self.y_pred_inverse), axis = 1)
        np.savetxt(pathsave + self.filenamesave + ".csv", t1, delimiter=",")


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
    def  bipolar_sigmoid_activation(x, c = 1.0):
        return (1.0 - np.exp(-x * c)) / (1.0 + np.exp(-x * c))
    
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
    

#pathsave = "/home/hunter/nguyenthieu95/ai/6_google_trace/SONIA/testing_new/cluster_pso/mutation/2m/result/cpu/"
#fullpath = "/home/hunter/nguyenthieu95/ai/data/GoogleTrace/"
#filename = "data_resource_usage_twoMinutes_6176858948.csv"

pathsave = "/home/thieunv/university/LabThayMinh/code/6_google_trace/SONIA/testing_new/cluster_pso/mutation/2m/result/cpu/"
fullpath = "/home/thieunv/university/LabThayMinh/code/data/GoogleTrace/"
filename = "data_resource_usage_twoMinutes_6176858948.csv"

df = read_csv(fullpath+ filename, header=None, index_col=False, usecols=[3], engine='python')   
dataset_original = df.values

sliding_windows = [3]    #[ 2, 3, 5]           # [3]  
list_num = [(12000, 20000)]
method_statistics = [2]
hidden_nodes = [8]

p_rts = [0.98 ]        # [0.9, 0.95, 0.99]        # Xac suat nhan biet
reper_sizes = [30]               # [2, 4, 6]                  # Repertoire size (population size)
g_maxs = [100]      
low_up_ws = [ [-1.0, 1.0] ]           # Lower and upper values for weights
low_up_bs = [ [-1.0, 1.0] ]          # Lower and upper values for biases


pl1 = 1         # Use to draw figure
#pl2 = 1000
so_vong_lap = 0

for list_idx in list_num:
    for sliding in sliding_windows:
        for method_statistic in method_statistics:
            for hidden_node in hidden_nodes:
                for p_rt in p_rts:
                    for reper_size in reper_sizes:
                        for g_max in g_maxs:
                            for low_up_w in low_up_ws:
                                for low_up_b in low_up_bs:
                                    
                                    ais_param = [p_rt, reper_size, g_max, low_up_w, low_up_b]
                                
                                    febpnn = Model(dataset_original, list_idx, hidden_node, ais_param, sliding, method_statistic)
                                    febpnn.fit()
                                                    
                                    so_vong_lap += 1
                                    if so_vong_lap % 1000 == 0:
                                        print "Vong lap thu : {0}".format(so_vong_lap)

print "Processing DONE !!!"
    
    