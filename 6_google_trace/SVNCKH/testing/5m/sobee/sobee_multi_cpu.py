import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../../")
from model import sobee1
from pandas import read_csv

# pathsave = "/home/thieunv/Desktop/Link to LabThayMinh/code/6_google_trace/SVNCKH/testing/3m/sopso/result/cpu_ram_cpu/"
# fullpath = "/home/thieunv/university/LabThayMinh/code/data/GoogleTrace/"

pathsave = "/home/hunter/nguyenthieu95/ai/6_google_trace/SVNCKH/testing/5m/sobee/result/multi_cpu/"
fullpath = "/home/hunter/nguyenthieu95/ai/data/GoogleTrace/"

filename3 = "data_resource_usage_3Minutes_6176858948.csv"
filename5 = "data_resource_usage_5Minutes_6176858948.csv"
filename8 = "data_resource_usage_8Minutes_6176858948.csv"
filename10 = "data_resource_usage_10Minutes_6176858948.csv"
df = read_csv(fullpath+ filename5, header=None, index_col=False, usecols=[3, 4], engine='python')
dataset_original = df.values

list_num3 = (11120, 13900, 0)
list_num5 = (6640, 8300, 0)
list_num8 = (4160, 5200, 0)
list_num10 = (3280, 4100, 0)

output_index = 0
method_statistic = 0
max_cluster=25
mutation_id=1
couple_activation = (2, 0)   # 0: elu, 1:relu, 2:tanh, 3:sigmoid

sliding_windows = [2, 5]  # [ 2, 3, 5]
positive_numbers = [0.25]  # [0.05, 0.15, 0.35]
stimulation_levels = [0.20]  # [0.10, 0.25, 0.45]
distance_levels = [0.75] # [0.65, 0.75, 0.85]

list_max_gens = [260]  # epoch
list_num_bees = [24, 30, 36]  # number of bees - population
num_sites = 3  # phan vung, 3 dia diem
elite_sites = 1
patch_size = 5.0
patch_factor = 0.97
e_bees = 10
o_bees = 3
low_up_w = [-1, 1]          # Lower and upper values for weights
low_up_b = [-1, 1]

fig_id = 1
so_vong_lap = 0
for sliding in sliding_windows:
    for pos_number in positive_numbers:
        for sti_level in stimulation_levels:
            for dist_level in distance_levels:

                for max_gens in list_max_gens:
                    for num_bees in list_num_bees:

                        para_data = {
                            "dataset": dataset_original,
                            "list_index": list_num5,
                            "output_index": output_index,
                            "method_statistic": method_statistic,
                            "sliding": sliding
                        }

                        para_net = {
                            "max_cluster": max_cluster, "pos_number": pos_number,
                            "sti_level": sti_level, "dist_level": dist_level,
                            "mutation_id": mutation_id, "couple_activation": couple_activation,
                            "path_save": pathsave, "fig_id": fig_id
                        }

                        para_bee = {
                            "max_gens": max_gens, "num_bees": num_bees,
                            "num_sites": num_sites, "elite_sites": elite_sites,
                            "patch_size": patch_size, "patch_factor": patch_factor,
                            "e_bees": e_bees, "o_bees": o_bees,
                            "lowup_w": low_up_b, "lowup_b": low_up_b
                        }

                        my_model = sobee1.Model(para_data, para_net, para_bee)
                        my_model.fit()
                        so_vong_lap += 1
                        fig_id += 2
                        if so_vong_lap % 5000 == 0:
                            print ("Vong lap thu : {0}".format(so_vong_lap))

print ("Processing DONE !!!")

