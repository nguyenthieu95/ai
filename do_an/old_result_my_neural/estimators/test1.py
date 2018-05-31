#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 01:40:43 2018

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
import copy
from random import randint, uniform, sample
from operator import add
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
        affinity = self.get_affinity(candidate)
        return [candidate, affinity]
    
    def repertoires(self, rs_size = 6):
        """ population
        rs_size: number of repertoire in antibody (population)
        """
        return [ self.create_candidate() for x in range(rs_size) ]
    
    def init_params(self):
        repertoires = self.repertoires(self.rs)
        print("Create repertoires done!!!")
        return repertoires
    
    def get_mae(self, reperoire, X_data, y_data):
        mae_list = []        
        self.t1 = self.size_w1 + self.size_b1
        self.t2 = self.size_w1 + self.size_b1 + self.size_w2
        
        w1 = np.reshape(reperoire[:self.size_w1], (self.number_node_input, -1))
        b1 = np.reshape(reperoire[self.size_w1:self.t1], (-1, self.size_b1))
        w2 = np.reshape(reperoire[self.t1:self.t2], (self.number_node_hidden, -1))
        b2 = np.reshape(reperoire[self.t2:], (-1, self.size_b2))  
        
        for i in range(0, len(X_data)):
            vector_hidden = np.add( np.matmul(X_data[i], w1), b1)
            input_hidden = Model.bipolar_sigmoid_activation(vector_hidden)
            output = np.add( np.matmul(input_hidden, w2), b2)
            y_pred = Model.elu_activation(output.flatten())
            mae_list.append( np.average( np.power( (y_data[i].flatten() - y_pred), 2 ) ) )
        temp = reduce(add, mae_list, 0)
        return temp / ( len(X_data) * 1.0)
    
    
    def get_affinity(self, reperoire):
        return -1 * sqrt(self.get_mae(reperoire, self.X_train, self.y_train))
    
    
    def get_ab_with_affinity_max(self, repertoires):
        temp = max(repertoires, key=lambda x: x[1])
        return copy.deepcopy(temp)    
    
    def get_prj(self, repe_max, current_repe):  # [repe_max, affinity], [current_repe, affinity]
        list_p = []
        max_rep = repe_max[0]
        cur_rep = current_repe[0]
        for i in range(len(max_rep)):
            t1 = abs( 1.0 - cur_rep[i] / max_rep[i] )
            t2 = 1.0 /exp(t1) if t1 < 100 else 0.0
            list_p.append(t2)
        temp = reduce(add, list_p, 0)
        return temp / len(max_rep)
    
    
    def somatic_hypermutation(self, antibody, current_generation):
        """
        2 task: uniform search and local fine-tuning
        antibody: [solution, affinity]
        """
        temp = copy.deepcopy(antibody[0])       
        for i in range(len(temp)):
            Ag = pow( (np.random.random_sample() * (1.0 - current_generation / self.g_max)), 2 )
            
            if (i < self.size_w1) or (self.t1 <= i and i < self.t2):
                low_up = self.low_up_w
            else:
                low_up = self.low_up_b

            if np.random.random_sample() < 0.5:
                temp[i] = temp[i] + Ag * (low_up[1] - temp[i])
            else:
                temp[i] = temp[i] - Ag * (temp[i] - low_up[0])
        affinity = self.get_affinity(temp)
        return [temp, affinity]
    
    def receptor_editing(self, antibody):
        temp = copy.deepcopy(antibody[0])
        temp = temp + np.reshape(pow( (np.random.random_sample()), 3.0 ) * np.random.standard_cauchy(len(temp)), (-1, 1))
        affinity = self.get_affinity(temp)
        return [temp, affinity]
        
    
    def introduce_diverse_abs(self, list_maturation):
        t = (self.rs - len(list_maturation)) / 2
        repertoires_new = []
        i = 0
        while(i < t):
            male = randint(0, len(list_maturation) - 1)   # father
            female = randint(0, len(list_maturation) - 1) # mother
            if male != female:
                father = copy.deepcopy(list_maturation[male][0])
                mother = copy.deepcopy(list_maturation[female][0])
                recomb_id = randint(0, len(father) - 1) 
                father[recomb_id] = mother[recomb_id] + np.random.random_sample(1)
                mother[recomb_id] = father[recomb_id] + np.random.random_sample(1)
                aff1 = self.get_affinity(father)
                aff2 = self.get_affinity(mother)
                repertoires_new.append( [father, aff1] )
                repertoires_new.append( [mother, aff2] )
                i += 1

        return repertoires_new + list_maturation
                
    
    def update_ab_repertoire(self, repertoires, repertoires_new):
        temp = repertoires + repertoires_new
        temp = sorted(temp, key=lambda x: x[1], reverse=True)
        return copy.deepcopy(temp[:self.rs])    

    
    def evaluate_test_accuracy(self, reper):
        self.get_mae(reper, self.X_test, self.y_test)
    
    def ais(self):
        # Step 1: Initialize parameters
        ### Training stage
        g = 0
        repertoires = self.init_params()    
        
        while g < self.g_max :
            ### Step 2: 
            ab_max = self.get_ab_with_affinity_max(repertoires)     # [solution, affinity]
            print("Epoch {0}, RMSE = {1}".format(g+1, pow(ab_max[1], 2.0) ))
            
            ### Step 3: Perform clonal selection
            list_promoted_ab = []
            list_unpromoted_ab = []
            for repe in repertoires:
                if self.get_prj(ab_max, repe) >= self.p_rt:
                    list_promoted_ab.append(copy.deepcopy(repe))
                else:
                    list_unpromoted_ab.append(copy.deepcopy(repe))
            
            ### Diverse abs
            v = 0.1
            number_lower = int ( v * len(list_unpromoted_ab))
            lower_index = sample(range(len(list_unpromoted_ab)), number_lower)
            for x in lower_index:
                list_promoted_ab.append( list_unpromoted_ab[x] )
            
            ### Step 4: Implement affinity maturation
            list_maturation = []
            for pro_ab in list_promoted_ab:
#                list_maturation.append(copy.deepcopy(pro_ab))
                if np.random.random_sample() <= 0.5:
                    list_maturation.append( self.somatic_hypermutation(copy.deepcopy(pro_ab), g) )
                else:
                    list_maturation.append( self.receptor_editing(copy.deepcopy(pro_ab)) )
            
            ### Step 5:
            repertoires_new = self.introduce_diverse_abs(copy.deepcopy(list_maturation))     # Chon tren tap trung gian
            
            ### Step 6: 
            self.update_ab_repertoire(repertoires, repertoires_new)
            
            g += 1
        
        #### Test Stage 
        ### Step 7:
        best = self.get_ab_with_affinity_max(repertoires)       # [solution, affinity]
        self.predict(best[0])
        
             
    def fit(self):
        self.preprocessing_data()
        self.ais()
        self.draw_predict()
        
    def predict(self, reperoire):
        w1 = np.reshape(reperoire[:self.size_w1], (self.number_node_input, -1))
        b1 = np.reshape(reperoire[self.size_w1:self.t1], (-1, self.size_b1))
        w2 = np.reshape(reperoire[self.t1:self.t2], (self.number_node_hidden, -1))
        b2 = np.reshape(reperoire[self.t2:], (-1, self.size_b2))  
        
        y_pred = []
        for i in range(0, len(self.X_test)):
            vector_hidden = np.add( np.matmul(self.X_test[i], w1), b1)
            input_hidden = Model.bipolar_sigmoid_activation(vector_hidden)
            output_hidden = np.add( np.matmul(input_hidden, w2), b2 )
            y_pred.append(Model.elu_activation(output_hidden.flatten()))
        
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
filename = "data_resource_usage_fiveMinutes_6176858948.csv"

df = read_csv(fullpath+ filename, header=None, index_col=False, usecols=[3, 4], engine='python')   
dataset_original = df.values

sliding_windows = [3]    #[ 2, 3, 5]           # [3]  
list_num = [(6400, 8000)]
method_statistics = [2]
hidden_nodes = [8]

p_rts = [0.3]        # [0.9, 0.95, 0.99]        # Xac suat nhan biet
reper_sizes = [100]               # [2, 4, 6]                  # Repertoire size (population size)
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
    