#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 11:13:18 2018

@author: thieunv
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 05:39:36 2018

@author: thieunv
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 23:29:35 2018

@author: thieunv
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 01:35:30 2018

@author: thieunv

THE ARTIFICIAL BEE COLONY ALGORITHM
IN TRAINING ARTIFICIAL NEURAL
NETWORK FOR OIL SPILL DETECTION

"""

import sys
sys.path.insert(0, '../')
import numpy as np
from scipy.spatial import distance
from math import exp, sqrt
from copy import deepcopy
from random import randint, uniform, random
from operator import add, itemgetter
from pandas import read_csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from preprocessing import TimeSeries


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
        self.X_train, self.y_train, self.X_test, self.y_test, self.min_max_scaler = TimeSeries.preprocessing_data(self.train_idx, 0, self.test_idx, self.sliding, self.method_statistic, self.dataset_original, self.min_max_scaler)
        print("Processing data done!!!")
    
    def create_search_space(self, low_up_w = None, low_up_b=None):   # [ [-1, 1], [-1, 1], ... ]
        self.number_node_input = self.X_train.shape[1]
        self.number_node_output = self.y_train.shape[1]
        
        self.size_w1 = self.number_node_input * self.number_node_hidden
        self.size_b1 = self.number_node_hidden
        self.size_w2 = self.number_node_hidden * self.number_node_output
        self.size_b2 = self.number_node_output
    
        self.t1 = self.size_w1 + self.size_b1
        self.t2 = self.size_w1 + self.size_b1 + self.size_w2
        
        w1 = [low_up_w for i in range(self.size_w1)]
        b1 = [low_up_b for i in range(self.size_b1)]
        w2 = [low_up_w for i in range(self.size_w2)]
        b2 = [low_up_b for i in range(self.size_b2)]
        search_space = w1 + b1 + w2 + b2
        return search_space
    
        
    def create_candidate(self, minmax):
        candidate = [ (minmax[i][1] - minmax[i][0]) * random() + minmax[i][0] for i in range(len(minmax))]
        return candidate
#        fitness = self.objective_function(candidate)
#        return [candidate, fitness]
    
    def get_mae(self, bee, X_data, y_data):
        mae_list = []        

        w1 = np.reshape(bee[:self.size_w1], (self.number_node_input, -1))
        b1 = np.reshape(bee[self.size_w1:self.t1], (-1, self.size_b1))
        w2 = np.reshape(bee[self.t1:self.t2], (self.number_node_hidden, -1))
        b2 = np.reshape(bee[self.t2:], (-1, self.size_b2))  
        
        for i in range(0, len(X_data)):
            vector_hidden = np.add( np.matmul(X_data[i], w1), b1)
#            input_hidden = Model.elu_activation(vector_hidden.flatten())
#            y_pred = Model.elu_activation(output.flatten())
            
            input_hidden = np.apply_along_axis(Model.elu_activation, 1, vector_hidden)
            output = np.add( np.matmul(input_hidden, w2), b2)
            y_pred = np.apply_along_axis(Model.elu_activation, 1, output)
            mae_list.append( np.average( np.power( (y_data[i].flatten() - y_pred), 2 ) ) )
        temp = reduce(add, mae_list, 0)
        return temp / ( len(X_data) * 1.0)
    
    
    def create_random_bee(self, search_space):
        return self.create_candidate(search_space)
    
    def objective_function(self, vector):
        return self.get_mae(vector, self.X_train, self.y_train)
    
    
    def create_neigh_bee(self, individual, patch_size, search_space):
        bee = []
        elem = 0.0
        for x in range(0, len(individual)):
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
    
    
    def search_neigh(self, parent, neigh_size, patch_size, search_space):  # parent:  [ vector_individual, fitness ]
        """
        Tim kiem trong so neigh_size, chi lay 1 hang xom tot nhat
        """
        neigh = [self.create_neigh_bee(parent[0], patch_size, search_space) for x in range(0, neigh_size)]
        neigh = [(bee, self.objective_function(bee)) for bee in neigh]
        neigh_sorted = sorted(neigh, key=itemgetter(1))
        return neigh_sorted[0]
    
    
    def create_scout_bees(self, search_space, num_scouts):  # So luong ong trinh tham
        return [self.create_random_bee(search_space) for x in range(0, num_scouts)]
    
    
    def search(self, max_gens, search_space, num_bees, num_sites, elite_sites, patch_size, e_bees, o_bees):
        pop = [self.create_random_bee(search_space) for x in range(0, num_bees)]
        for j in range(0, max_gens):
            pop = [(bee, self.objective_function(bee)) for bee in pop]
            pop_sorted = sorted(pop, key=itemgetter(1))
            best = pop_sorted[0]
    
            next_gen = []
            for i in range(0, num_sites):
                if i < elite_sites:
                    neigh_size = e_bees
                else:
                    neigh_size = o_bees
                next_gen.append(self.search_neigh(pop_sorted[i], neigh_size, patch_size, search_space))
    
            scouts = self.create_scout_bees(search_space, (num_bees - num_sites))  # Ong trinh tham
            pop = [x[0] for x in next_gen] + scouts
            patch_size = patch_size * 0.99
            print("Epoch = {0}, patch_size = {1}, best = {2}".format(j + 1, patch_size, best[1]))
        return best

    def main_abc(self):
        search_space = self.create_search_space(self.low_up_w, self.low_up_b)
        
        max_gens = 100  # epoch
        num_bees = 45  # number of bees - population
        num_sites = 3  # phan vung, 3 dia diem 
        elite_sites = 1
        patch_size = 3.0
        e_bees = 7
        o_bees = 2
    
        best = self.search(max_gens, search_space, num_bees, num_sites, elite_sites, patch_size, e_bees, o_bees)
        print("done! Solution: f = {0}, s = {1}".format(best[1], best[0]))
    
        #### Test Stage 
        self.predict(best)
        
             
    def fit(self):
        self.preprocessing_data()
        self.main_abc()
        self.draw_predict()
        
    def predict(self, bee):
        w1 = np.reshape(bee[:self.size_w1], (self.number_node_input, -1))
        b1 = np.reshape(bee[self.size_w1:self.t1], (-1, self.size_b1))
        w2 = np.reshape(bee[self.t1:self.t2], (self.number_node_hidden, -1))
        b2 = np.reshape(bee[self.t2:], (-1, self.size_b2))  
        
        y_pred = []
        for i in range(0, len(self.X_test)):
            
            w1 = np.reshape(bee[:self.size_w1], (self.number_node_input, -1))
            b1 = np.reshape(bee[self.size_w1:self.t1], (-1, self.size_b1))
            w2 = np.reshape(bee[self.t1:self.t2], (self.number_node_hidden, -1))
            b2 = np.reshape(bee[self.t2:], (-1, self.size_b2))  
        
            vector_hidden = np.add( np.matmul(self.X_test[i], w1), b1)
            input_hidden = np.apply_along_axis(Model.elu_activation, 1, vector_hidden)
            output = np.add( np.matmul(input_hidden, w2), b2)
            y_pred.append(Model.elu_activation(output.flatten()))
        
        # Evaluate models on the test set
        y_test_inverse = self.min_max_scaler.inverse_transform(self.y_test)
        y_pred_inverse = self.min_max_scaler.inverse_transform(np.array(y_pred).reshape(-1, self.dimension))
        
        testScoreRMSE = sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
        testScoreMAE = mean_absolute_error(y_test_inverse, y_pred_inverse)
        
        self.y_predict, self.score_test_RMSE, self.score_test_MAE = y_pred, testScoreRMSE, testScoreMAE
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
        plt.plot(self.y_test_inverse[:, :1])
        plt.plot(self.y_pred_inverse[:, :1])
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
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def elu_activation(x):
#        return (np.exp(x) - 1.0) if x < 0 else x 
        return np.array( [np.exp(i) - 1.0 if i < 0 else i for i in x] )
        
    @staticmethod
    def relu_activation(x):
        return np.max(x, 0)
    
    @staticmethod 
    def tanh_activation(x):
        return np.tanh(x)
    
    @staticmethod
    def bipolar_sigmoid_activation(x, c = 1.0):
        return (1.0 - np.exp(-x * c)) / (1.0 + np.exp(-x * c))
    
    @staticmethod
    def get_random_input_vector(train_X):
        return deepcopy(train_X[randint(0, len(train_X)-1)])
    
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
filename = "data_resource_usage_fiveMinutes_6176858948.csv"

df = read_csv(fullpath+ filename, header=None, index_col=False, usecols=[3, 4], engine='python')   
dataset_original = df.values

sliding_windows = [3]    #[ 2, 3, 5]           # [3]  
list_num = [(6400, 8000)]
method_statistics = [2]
hidden_nodes = [8]

p_rts = [0.9 ]        # [0.9, 0.95, 0.99]        # Xac suat nhan biet
reper_sizes = [32]               # [2, 4, 6]                  # Repertoire size (population size)
g_maxs = [10]      
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
    