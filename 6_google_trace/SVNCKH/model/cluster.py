from math import ceil, exp
from copy import deepcopy
from operator import itemgetter
import numpy as np
from sklearn.cluster import KMeans

from utils import MathHelper

# t = MathHelper.distance_func([12, 3], [3, 4])


class Clustering(object):
    def __init__(self, stimulation_level=0.15, positive_number=0.15, max_cluster=20,
                 neighbourhood_density=0.2, gauss_width=1.0,
                 number_cluster=8, random_state=0,
                 distance_level=0.25, mutation_id=1, activation_id=2, dataset=None):
        """
        sobee: st, lr, mc, nd, gw
        sonia: st, lr, pn, mc

        density <= 0.5 , gauss_width = [0.2, 0.5, 1.0, 5.0] (phuong sai)

        0, 1, 2, 3: elu, relu, tanh, sigmoid
        """
        self.stimulation_level = stimulation_level
        self.max_cluster = max_cluster

        self.neighbourhood_density = neighbourhood_density
        self.gauss_width = gauss_width

        self.positive_number = positive_number

        self.number_cluster = number_cluster
        self.random_state = random_state

        self.distance_level = distance_level
        self.mutation_id = mutation_id
        self.activation_id = activation_id
        self.dataset = dataset
        self.dimension = dataset.shape[1]



    def sobee(self, dataset=None):
        ### Qua trinh train va dong thoi tao cac hidden unit (Pha 1 - cluster data)
        # 2. Khoi tao hidden thu 1
        hu1 = [0, MathHelper.get_random_input_vector(dataset)]  # hidden unit 1 (t1, wH)
        list_clusters = [deepcopy(hu1)]  # list hidden units
        centers = deepcopy(hu1[1]).reshape(1, hu1[1].shape[0])  # Mang 2 chieu
        #    training_detail_file_name = full_path + 'SL=' + str(stimulation_level) + '_Slid=' + str(sliding) + '_Epoch=' + str(epoch) + '_BS=' + str(batch_size) + '_LR=' + str(learning_rate) + '_PN=' + str(positive_number) + '_CreateHU.txt'
        m = 0
        while m < len(dataset):
            list_dist_mj = []  # Danh sach cac dist(mj)
            # number of hidden units
            for j in range(0, len(list_clusters)):  # j: la chi so cua hidden thu j
                list_dist_mj.append([j, MathHelper.distance_func(dataset[m], centers[j])])
            list_dist_mj = sorted(list_dist_mj, key=itemgetter(1))  # Sap xep tu be den lon

            c = list_dist_mj[0][0]  # c: Chi so (index) cua hidden unit thu c ma dat khoang cach min
            distmc = list_dist_mj[0][1]  # distmc: Gia tri khoang cach nho nhat

            if distmc < self.stimulation_level:
                list_clusters[c][0] += 1  # update hidden unit cth

                # Find Neighbourhood
                list_distjc = []
                for i in range(0, len(centers)):
                    list_distjc.append([i, MathHelper.distance_func(centers[c], centers[i])])
                list_distjc = sorted(list_distjc, key=itemgetter(1))

                # Update BMU (Best matching unit and it's neighbourhood)
                neighbourhood_node = int(1 + ceil(self.neighbourhood_density * (len(list_clusters) - 1)))
                for i in range(0, neighbourhood_node):
                    if i == 0:
                        centers[c] += (self.positive_number * distmc) * (dataset[m] - list_clusters[c][1])
                        list_clusters[c][1] += (self.positive_number * distmc) * (dataset[m] - list_clusters[c][1])
                    else:
                        c_temp, distjc = list_distjc[i][0], list_distjc[i][1]
                        hic = exp(-distjc * distjc / self.gauss_width)
                        delta = (self.positive_number * hic) * (dataset[m] - list_clusters[c_temp][1])

                        list_clusters[c_temp][1] += delta
                        centers[c_temp] += delta
                # Tiep tuc vs cac example khac
                m += 1
                # if m % 1000 == 0:
                #     print "distmc = {0}".format(distmc)
                #     print "m = {0}".format(m)
            else:
                # print "Failed !!!. distmc = {0}".format(distmc)
                list_clusters.append([0, deepcopy(dataset[m])])
                # print "Hidden unit thu: {0} duoc tao ra.".format(len(list_clusters))
                centers = np.append(centers, [deepcopy(dataset[m])], axis=0)
                for hu in list_clusters:
                    hu[0] = 0
                # then go to step 1
                m = 0
                if len(list_clusters) > self.max_cluster:
                    break
                    ### +++
        self.count_centers = len(list_clusters)
        self.centers = deepcopy(centers)
        self.list_clusters = deepcopy(list_clusters)


    def sonia(self, dataset=None):
        ### Qua trinh train va dong thoi tao cac hidden unit (Pha 1 - cluster data)
        # 2. Khoi tao hidden thu 1
        hu1 = [0, MathHelper.get_random_input_vector(dataset)]  # hidden unit 1 (t1, wH)
        list_clusters = [deepcopy(hu1)]  # list hidden units
        centers = deepcopy(hu1[1]).reshape(1, hu1[1].shape[0])  # Mang 2 chieu
        #    training_detail_file_name = full_path + 'SL=' + str(stimulation_level) + '_Slid=' + str(sliding) + '_Epoch=' + str(epoch) + '_BS=' + str(batch_size) + '_LR=' + str(learning_rate) + '_PN=' + str(positive_number) + '_CreateHU.txt'
        m = 0
        while m < len(dataset):
            list_dist_mj = []  # Danh sach cac dist(mj)
            # number of hidden units
            for j in range(0, len(list_clusters)):  # j: la chi so cua hidden thu j
                list_dist_mj.append([j, MathHelper.distance_func(dataset[m], centers[j])])
            list_dist_mj = sorted(list_dist_mj, key=itemgetter(1))  # Sap xep tu be den lon

            c = list_dist_mj[0][0]  # c: Chi so (index) cua hidden unit thu c ma dat khoang cach min
            distmc = list_dist_mj[0][1]  # distmc: Gia tri khoang cach nho nhat

            if distmc < self.stimulation_level:
                list_clusters[c][0] += 1  # update hidden unit cth

                for i in range(len(list_clusters)):
                    centers[i] += self.positive_number * distmc * (dataset[m] - list_clusters[i][1])
                    list_clusters[i][1] += self.positive_number * distmc * (dataset[m] - list_clusters[i][1])
                # Tiep tuc vs cac example khac
                m += 1
                # if m % 1000 == 0:
                #     print "distmc = {0}".format(distmc)
                #     print "m = {0}".format(m)
            else:
                #                print "Failed !!!. distmc = {0}".format(distmc)
                list_clusters.append([0, deepcopy(dataset[m])])
                # print "Hidden unit thu: {0} duoc tao ra.".format(len(list_clusters))
                centers = np.append(centers, [deepcopy(dataset[m])], axis=0)
                for hu in list_clusters:
                    hu[0] = 0
                # then go to step 1
                m = 0
                if len(list_clusters) > self.max_cluster:
                    break
                    ### +++

        self.count_centers = len(list_clusters)
        self.centers = deepcopy(centers)
        self.list_clusters = deepcopy(list_clusters)



    def kmeans(self, dataset=None):
        kmeans = KMeans(n_clusters=self.number_cluster, random_state=self.random_state).fit(dataset)
        labelX = kmeans.predict(dataset).tolist()
        centers = kmeans.cluster_centers_

        list_clusters = []
        for i in range(len(centers)):
            temp = labelX.count(i)
            list_clusters.append([temp, centers[i]])

        self.count_centers = len(list_clusters)
        self.centers = deepcopy(centers)
        self.list_clusters = deepcopy(list_clusters)


    def mutation_cluster(self, dataset=None):
        self.threshold_number = int(len(dataset) / len(self.list_clusters))
        ### Qua trinh mutated hidden unit (Pha 2- Adding artificial local data)
        # Adding 2 hidden unit in begining and ending points of input space
        t1 = np.zeros(self.dimension)
        t2 = np.ones(self.dimension)

        self.list_clusters.append([0, t1])
        self.list_clusters.append([0, t2])
        self.centers = np.concatenate((self.centers, np.array([t1])), axis=0)
        self.centers = np.concatenate((self.centers, np.array([t2])), axis=0)

        #    # Sort matrix weights input and hidden, Sort list hidden unit by list weights
        for i in range(0, self.centers.shape[1]):
            self.centers = sorted(self.centers, key=lambda elem_list: elem_list[i])
            self.list_clusters = sorted(self.list_clusters, key=lambda elem_list: elem_list[1][i])

            for i in range(len(self.list_clusters) - 1):
                ta, wHa = self.list_clusters[i][0], self.list_clusters[i][1]
                tb, wHb = self.list_clusters[i + 1][0], self.list_clusters[i + 1][1]
                dab = MathHelper.distance_func(wHa, wHb)

                if dab > self.distance_level and ta < self.threshold_number and tb < self.threshold_number:
                    # Create new mutated hidden unit (Dot Bien)
                    temp_node = MathHelper.get_mutate_vector_weight(wHa, wHb, self.mutation_id)
                    self.list_clusters.insert(i + 1, [0, deepcopy(temp_node)])
                    self.centers = np.insert(self.centers, [i + 1], deepcopy(temp_node), axis=0)
            #         print "New hidden unit created. {0}".format(len(self.matrix_Wih))
            # print("Finished mutation hidden unit!!!")


    def sobee_without_mutation(self):
        self.sobee(self.dataset)
        return self.centers, self.list_clusters, self.count_centers

    def sobee_with_mutation(self):
        self.sobee(self.dataset)
        self.mutation_cluster(self.dataset)
        return self.centers, self.list_clusters, self.count_centers


    def sonia_without_mutation(self):
        self.sonia(self.dataset)
        return self.centers, self.list_clusters, self.count_centers

    def sonia_with_mutation(self):
        self.sonia(self.dataset)
        self.mutation_cluster(self.dataset)
        return self.centers, self.list_clusters, self.count_centers


    def kmeans_without_mutation(self):
        self.kmeans(self.dataset)
        return self.centers, self.list_clusters, self.count_centers

    def kmeans_with_mutation(self):
        self.kmeans(self.dataset)
        self.mutation_cluster(self.dataset)
        return self.centers, self.list_clusters, self.count_centers


    def transform_features(self, features=None):
        temp = []
        if self.activation_id == 0:
            for i in range(0, len(features)):
                Sih = []
                for j in range(0, len(self.centers)):  # (w11, w21) (w12, w22), (w13, w23)
                    Sih.append(MathHelper.elu(MathHelper.distance_func(self.centers[j], features[i])))
                temp.append(np.array(Sih))
        elif self.activation_id == 1:
            for i in range(0, len(features)):
                Sih = []
                for j in range(0, len(self.centers)):  # (w11, w21) (w12, w22), (w13, w23)
                    Sih.append(MathHelper.relu(MathHelper.distance_func(self.centers[j], features[i])))
                temp.append(np.array(Sih))
        elif self.activation_id == 2:
            for i in range(0, len(features)):
                Sih = []
                for j in range(0, len(self.centers)):  # (w11, w21) (w12, w22), (w13, w23)
                    Sih.append(MathHelper.tanh(MathHelper.distance_func(self.centers[j], features[i])))
                temp.append(np.array(Sih))
        else:
            for i in range(0, len(features)):
                Sih = []
                for j in range(0, len(self.centers)):  # (w11, w21) (w12, w22), (w13, w23)
                    Sih.append(MathHelper.sigmoid(MathHelper.distance_func(self.centers[j], features[i])))
                temp.append(np.array(Sih))
        return np.array(temp)


