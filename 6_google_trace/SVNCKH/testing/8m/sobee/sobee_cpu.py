import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../../")
from model import sobee
from pandas import read_csv

# pathsave = "/home/thieunv/Desktop/Link to LabThayMinh/code/6_google_trace/SVNCKH/testing/3m/sopso/result/cpu_ram_cpu/"
# fullpath = "/home/thieunv/university/LabThayMinh/code/data/GoogleTrace/"

pathsave = "/home/hunter/nguyenthieu95/ai/6_google_trace/SVNCKH/testing/8m/sobee/result/cpu/"
fullpath = "/home/hunter/nguyenthieu95/ai/data/GoogleTrace/"

filename8 = "data_resource_usage_8Minutes_6176858948.csv"
df = read_csv(fullpath+ filename8, header=None, index_col=False, usecols=[3], engine='python')
dataset_original = df.values

list_num8 = (4160, 5200, 0)
output_index = 0
method_statistic = 0
max_cluster=20
neighbourhood_density=0.2
gauss_width=1.0
mutation_id=1
couple_acti = (0, 0)           # 0: elu, 1:relu, 2:tanh, 3:sigmoid

sliding_windows = [ 2, 3, 5]
positive_numbers =  [0.05, 0.15, 0.35]
stimulation_levels = [0.15, 0.25, 0.45]
distance_levels = [0.55, 0.70, 0.85]


list_max_gens = [180, 320, 520]
list_num_bees = [16, 36, 52]                # number of bees - population
num_sites = 3                               # phan vung, 3 dia diem
elite_sites = 1
patch_size = 5.0
patch_factor = 0.97
e_bees = 7
o_bees = 3
low_up_w = [-0.2, 0.6]                      # Lower and upper values for weights
low_up_b = [-0.5, 0.5]


fig_id = 1
so_vong_lap = 0
for sliding in sliding_windows:
    for pos_number in positive_numbers:
        for sti_level in stimulation_levels:
            for dist_level in distance_levels:

                for max_gens in list_max_gens:
                    for num_bees in list_num_bees:

                        my_model = sobee.Model(dataset_original, list_num8, output_index, sliding, method_statistic, max_cluster,
                                               pos_number, sti_level, dist_level, mutation_id, couple_acti, fig_id, pathsave,
                                               max_gens, num_bees, num_sites, elite_sites, patch_size, patch_factor, e_bees, o_bees, low_up_w, low_up_b)
                        my_model.fit()
                        so_vong_lap += 1
                        fig_id += 2
                        if so_vong_lap % 100 == 0:
                                print "Vong lap thu : {0}".format(so_vong_lap)
print "Processing DONE !!!"

