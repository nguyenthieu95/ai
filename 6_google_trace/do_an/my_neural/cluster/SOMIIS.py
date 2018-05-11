#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 14:41:27 2018

@author: thieunv

 Self-Organizing Map Inspired by the Immune System
"""

import sys
sys.path.insert(0, '../')

from math import ceil, exp
from copy import deepcopy
from operator import itemgetter
import numpy as np
import MathHelper
#t = MathHelper.distance_func([12, 3], [3, 4])


class Cluster(object):
    def __init__(self, stimulation_level=0.15, learning_rate=0.15, max_cluster=20, neighbourhood_density=0.2, gauss_width=1.0):
        """
        density <= 0.5 , gauss_width = [0.2, 0.5, 1.0, 5.0] (phuong sai)
        """
        self.stimulation_level = stimulation_level
        self.learning_rate = learning_rate
        self.max_cluster = 20
        self.neighbourhood_density = neighbourhood_density
        self.gauss_width = gauss_width

    def clustering(self, dataset=None):
        ### Qua trinh train va dong thoi tao cac hidden unit (Pha 1 - cluster data)
        # 2. Khoi tao hidden thu 1
        hu1 = [0, MathHelper.get_random_input_vector(dataset)]   # hidden unit 1 (t1, wH)
        list_clusters = [deepcopy(hu1)]         # list hidden units
        centers = deepcopy(hu1[1]).reshape(1, hu1[1].shape[0])     # Mang 2 chieu 
        #    training_detail_file_name = full_path + 'SL=' + str(stimulation_level) + '_Slid=' + str(sliding) + '_Epoch=' + str(epoch) + '_BS=' + str(batch_size) + '_LR=' + str(learning_rate) + '_PN=' + str(positive_number) + '_CreateHU.txt'
        m = 0
        while m < len(dataset):
            list_dist_mj = []      # Danh sach cac dist(mj)
             # number of hidden units
            for j in range(0, len(list_clusters)):                # j: la chi so cua hidden thu j
                list_dist_mj.append([j, MathHelper.distance_func(dataset[m], centers[j])])
            list_dist_mj = sorted(list_dist_mj, key=itemgetter(1))        # Sap xep tu be den lon
            
            c = list_dist_mj[0][0]      # c: Chi so (index) cua hidden unit thu c ma dat khoang cach min
            distmc = list_dist_mj[0][1] # distmc: Gia tri khoang cach nho nhat
            
            if distmc < self.stimulation_level:
                list_clusters[c][0] += 1                  # update hidden unit cth
                
                # Find Neighbourhood
                list_distjc = []
                for i in range(0, len(centers)):
                    list_distjc.append([i, MathHelper.distance_func(centers[c], centers[i])])
                list_distjc = sorted(list_distjc, key=itemgetter(1))
                
                # Update BMU (Best matching unit and it's neighbourhood)
                neighbourhood_node = int( 1 + ceil( self.neighbourhood_density * (len(list_clusters) - 1) ) )
                for i in range(0, neighbourhood_node ):
                    if i == 0:
                        list_clusters[c][1] += (self.learning_rate * distmc) * (dataset[m] - list_clusters[c][1])
                        centers[c] += (self.learning_rate * distmc) * (dataset[m] - list_clusters[c][1])
                    else:
                        c_temp, distjc = list_distjc[i][0], list_distjc[i][1]
                        hic = exp(-distjc * distjc / self.gauss_width)
                        delta = (self.learning_rate * hic) * (dataset[m] - list_clusters[c_temp][1])
                        
                        list_clusters[c_temp][1] += delta
                        centers[c_temp] += delta
                # Tiep tuc vs cac example khac
                m += 1
        #                if m % 1000 == 0:
        #                    print "distmc = {0}".format(distmc)
        #                    print "m = {0}".format(m)
            else:
        #                print "Failed !!!. distmc = {0}".format(distmc)
                list_clusters.append([0, deepcopy(dataset[m]) ])
        #                print "Hidden unit thu: {0} duoc tao ra.".format(len(list_clusters))
                centers = np.append(centers, [deepcopy(dataset[m])], axis = 0)
                for hu in list_clusters:
                    hu[0] = 0
                # then go to step 1
                m = 0
                if len(list_clusters) > self.max_cluster:
                    break
                ### +++
        self.centers = deepcopy(centers)
        self.list_clusters = deepcopy(list_clusters)

    def mutation_cluster(self, dataset=None, distance_level=0.25, mutation_id=1):
        self.threshold_number = int (len(dataset) / len(self.list_clusters))
        ### Qua trinh mutated hidden unit (Pha 2- Adding artificial local data)
        # Adding 2 hidden unit in begining and ending points of input space
        t1 = np.zeros(dataset.shape[1])
        t2 = np.ones(dataset.shape[1])
        self.list_clusters.append([0, t1])
        self.list_clusters.append([0, t2])
        self.centers = np.concatenate((self.centers, np.array([t1])), axis=0)
        self.centers = np.concatenate((self.centers, np.array([t2])), axis=0)
    
    #    # Sort matrix weights input and hidden, Sort list hidden unit by list weights
        for i in range(0, self.matrix_Wih.shape[1]):
            self.centers = sorted(self.centers, key=lambda elem_list: elem_list[i])
            self.list_clusters = sorted(self.list_clusters, key=lambda elem_list: elem_list[1][i])
             
            for i in range(len(self.list_clusters) - 1):
                ta, wHa = self.list_clusters[i][0], self.list_clusters[i][1]
                tb, wHb = self.list_clusters[i+1][0], self.list_clusters[i+1][1]
                dab = MathHelper.distance_func(wHa, wHb)
                
                if dab > distance_level and ta < self.threshold_number and tb < self.threshold_number:
                    # Create new mutated hidden unit (Dot Bien)
                    temp_node = MathHelper.get_mutate_vector_weight(wHa, wHb, mutation_id)
                    self.list_clusters.insert(i+1, [0, deepcopy(temp_node)])
                    self.centers = np.insert(self.centers, [i+1], deepcopy(temp_node), axis=0)
#                    print "New hidden unit created. {0}".format(len(self.matrix_Wih))
#        print("Finished mutation hidden unit!!!")

    def cluster_without_mutation(self, dataset):
        self.clustering(dataset)
        return self.centers, self.list_clusters
    
    def cluster_with_mutation(self, dataset=None, distance_level=None, mutation_id=None):
        self.clustering(dataset)
        self.mutation_cluster(dataset, distance_level, mutation_id)
        return self.centers, self.list_clusters

    def transform_features(self, features=None, activation_id=2):
        """
        0, 1, 2, 3: elu, relu, tanh, sigmoid
        """
        temp = []
        if activation_id == 0:
            for i in range(0, len(features)):  
                Sih = []
                for j in range(0, len(self.centers)):     # (w11, w21) (w12, w22), (w13, w23)
                    Sih.append(MathHelper.elu( MathHelper.distance_func(self.centers[j], features[i])))
                temp.append(np.array(Sih))
        elif activation_id == 1:
            for i in range(0, len(features)):  
                Sih = []
                for j in range(0, len(self.centers)):     # (w11, w21) (w12, w22), (w13, w23)
                    Sih.append(MathHelper.relu( MathHelper.distance_func(self.centers[j], features[i])))
                temp.append(np.array(Sih))
        elif activation_id == 2:
            for i in range(0, len(features)):  
                Sih = []
                for j in range(0, len(self.centers)):     # (w11, w21) (w12, w22), (w13, w23)
                    Sih.append(MathHelper.tanh( MathHelper.distance_func(self.centers[j], features[i])))
                temp.append(np.array(Sih))
        else:
            for i in range(0, len(features)):  
                Sih = []
                for j in range(0, len(self.centers)):     # (w11, w21) (w12, w22), (w13, w23)
                    Sih.append(MathHelper.sigmoid( MathHelper.distance_func(self.centers[j], features[i])))
                temp.append(np.array(Sih))
        self.transformed_data = np.array(temp)
    

















