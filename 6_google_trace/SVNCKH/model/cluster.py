from math import ceil, exp
from copy import deepcopy
from operator import itemgetter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import MathHelper

# t = MathHelper.distance_func([12, 3], [3, 4])


class Clustering(object):
    def __init__(self, stimulation_level=None, positive_number=None, max_cluster=None,
                 neighbourhood_density=None, gauss_width=None,
                 number_cluster=None, random_state=None,
                 distance_level=None, mutation_id=None, activation_id=None, dataset=None):
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
        self.dataset = dataset
        self.dimension = dataset.shape[1]
        if activation_id == 0:
            self.activation = MathHelper.elu
        elif activation_id == 1:
            self.activation = MathHelper.relu
        elif activation_id == 2:
            self.activation = MathHelper.tanh
        else:
            self.activation = MathHelper.sigmoid


    def sobee(self, dataset=None):
        ### Qua trinh train va dong thoi tao cac hidden unit (Pha 1 - cluster data)
        # 2. Khoi tao hidden thu 1
        y = np.zeros(len(dataset))
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
                y[m] = c
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
                        centers[c] = centers[c] + self.positive_number * distmc * (dataset[m] - list_clusters[c][1])
                        list_clusters[c][1] = list_clusters[c][1] + self.positive_number * distmc * (dataset[m] - list_clusters[c][1])
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
        self.y = deepcopy(y)
        self.count_centers = len(list_clusters)
        self.centers_old = deepcopy(centers)
        self.centers = deepcopy(centers)
        self.list_clusters = deepcopy(list_clusters)


    def sonia(self, dataset=None):
        ### Qua trinh train va dong thoi tao cac hidden unit (Pha 1 - cluster data)
        # 2. Khoi tao hidden thu 1
        y = np.zeros(len(dataset))
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
                y[m] = c
                list_clusters[c][0] += 1  # update hidden unit cth

                centers[c] = centers[c] + self.positive_number * distmc * (dataset[m] - list_clusters[c][1])
                list_clusters[c][1] = list_clusters[c][1] + self.positive_number * distmc * (dataset[m] - list_clusters[c][1])

                # for i in range(len(list_clusters)):
                #     centers[i] = centers[i] + self.positive_number * distmc * (dataset[m] - list_clusters[i][1])
                #     list_clusters[i][1] = list_clusters[i][1] + self.positive_number * distmc * (dataset[m] - list_clusters[i][1])

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

        self.y = deepcopy(y)
        self.count_centers = len(list_clusters)
        self.centers_old = deepcopy(centers)
        self.centers = deepcopy(centers)
        self.list_clusters = deepcopy(list_clusters)



    def kmeans(self, dataset=None):
        kmeans = KMeans(n_clusters=self.number_cluster, random_state=self.random_state).fit(dataset)
        labelX = kmeans.predict(dataset)
        labelX_temp = labelX.tolist()
        centers = kmeans.cluster_centers_

        list_clusters = []
        for i in range(len(centers)):
            temp = labelX_temp.count(i)
            list_clusters.append([temp, centers[i]])

        self.y = deepcopy(labelX)
        self.count_centers = len(list_clusters)
        self.centers_old = deepcopy(centers)
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


    def calculate_silhouette_score(self, model_name = None, dataset=None):
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(dataset) + (self.count_centers + 1) * 50])

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(dataset, self.y)
        print("For n_clusters =", self.count_centers, "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(dataset, self.y)

        y_lower = 10
        for i in range(self.count_centers):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[self.y == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.get_cmap("Spectral")(float(i) / self.count_centers)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title(".") # The silhouette plot for the various clusters.
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = plt.get_cmap("Spectral")(self.y.astype(float) / self.count_centers)
        ax2.scatter(dataset[:, 0], dataset[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

        # Labeling the clusters
        centers = np.reshape(self.centers_old, (-1, 2))
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

        ax2.set_title(".")#The visualization of the clustered data.
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for " + model_name + " clustering with n_clusters = %d" % self.count_centers), fontsize=14, fontweight='bold')

        plt.show()

    def sobee_without_mutation(self):
        self.sobee(self.dataset)
        # self.calculate_silhouette_score("SoBee", self.dataset)
        return self.centers, self.list_clusters, self.count_centers, self.y

    def sobee_with_mutation(self):
        self.sobee(self.dataset)
        # self.calculate_silhouette_score("SoBee", self.dataset)
        self.mutation_cluster(self.dataset)
        return self.centers, self.list_clusters, self.count_centers, self.y


    def sonia_without_mutation(self):
        self.sonia(self.dataset)
        # self.calculate_silhouette_score("SONIA", self.dataset)
        return self.centers, self.list_clusters, self.count_centers, self.y

    def sonia_with_mutation(self):
        self.sonia(self.dataset)
        self.mutation_cluster(self.dataset)
        # self.calculate_silhouette_score("SONIA", self.dataset)
        return self.centers, self.list_clusters, self.count_centers, self.y


    def kmeans_without_mutation(self):
        self.kmeans(self.dataset)
        # self.calculate_silhouette_score("SONIA", self.dataset)
        return self.centers, self.list_clusters, self.count_centers, self.y

    def kmeans_with_mutation(self):
        self.sonia(self.dataset)
        self.number_cluster = self.count_centers
        self.kmeans(self.dataset)
        self.mutation_cluster(self.dataset)
        # self.calculate_silhouette_score(self.dataset)
        return self.centers, self.list_clusters, self.count_centers, self.y


    def sobee_new_no_mutation(self):
        self.sonia(self.dataset)
        self.number_cluster = self.count_centers
        self.kmeans(self.dataset)
        # self.calculate_silhouette_score("SoBee", self.dataset)
        return self.centers, self.list_clusters, self.count_centers, self.y

    def sobee_new_with_mutation(self):
        self.sonia(self.dataset)
        self.number_cluster = self.count_centers
        self.kmeans(self.dataset)
        # self.calculate_silhouette_score("SoBee", self.dataset)
        self.mutation_cluster(self.dataset)
        return self.centers, self.list_clusters, self.count_centers, self.y

    def transform_features(self, features=None):
        temp = []
        for i in range(0, len(features)):
            Sih = []
            for j in range(0, len(self.centers)):  # (w11, w21) (w12, w22), (w13, w23)
                Sih.append(self.activation(MathHelper.distance_func(self.centers[j], features[i])))
            temp.append(np.array(Sih))
        return np.array(temp)


