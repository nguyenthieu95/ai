import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../../")
from model import script1
from pandas import read_csv

# pathsave = "/home/thieunv/Desktop/Link to LabThayMinh/code/6_google_trace/SVNCKH/testing/3m/sonia/result/cpu/"
# fullpath = "/home/thieunv/university/LabThayMinh/code/data/GoogleTrace/"

pathsave = "/home/hunter/nguyenthieu95/ai/6_google_trace/SVNCKH/script1/10m/sonia/result/ram/"
fullpath = "/home/hunter/nguyenthieu95/ai/data/GoogleTrace/"

filename3 = "data_resource_usage_3Minutes_6176858948.csv"
filename5 = "data_resource_usage_5Minutes_6176858948.csv"
filename8 = "data_resource_usage_8Minutes_6176858948.csv"
filename10 = "data_resource_usage_10Minutes_6176858948.csv"
df = read_csv(fullpath+ filename10, header=None, index_col=False, usecols=[4], engine='python')
dataset_original = df.values

list_num3 = (11120, 13900, 0)
list_num5 = (6640, 8300, 0)
list_num8 = (4160, 5200, 0)
list_num10 = (3280, 4100, 0)

output_index = 0                # 0: cpu, 1: ram
method_statistic = 0
max_cluster=30
mutation_id=1
couple_activation = (2, 0)        # 0: elu, 1:relu, 2:tanh, 3:sigmoid

model = 0       # 0: sonia, 1: sobee

epochs = [2000]
batch_sizes = [32]
learning_rates = [0.15]
sliding_windows = [ 2, 5]
positive_numbers = [0.05, 0.15, 0.35]
stimulation_levels = [0.20, 0.30, 0.40, 0.50]
distance_levels = [0.65]

fig_id = 1
so_vong_lap = 0
for sliding in sliding_windows:
    for pos_number in positive_numbers:
        for sti_level in stimulation_levels:
            for dist_level in distance_levels:

                for epoch in epochs:
                    for batch_size in batch_sizes:
                        for learning_rate in learning_rates:

                            para_data = {
                                "dataset": dataset_original,
                                "list_index": list_num10,
                                "output_index": output_index,
                                "method_statistic": method_statistic,
                                "sliding": sliding
                            }

                            para_net = {
                                "model": model, "epoch": epoch, "batch_size": batch_size, "learning_rate": learning_rate,
                                "max_cluster": max_cluster, "pos_number": pos_number,
                                "sti_level": sti_level, "dist_level": dist_level,
                                "mutation_id": mutation_id, "couple_activation": couple_activation,
                                "path_save": pathsave, "fig_id": fig_id
                            }

                            my_model = script1.Model(para_data, para_net)
                            my_model.fit()
                            so_vong_lap += 1
                            fig_id += 2
                            if so_vong_lap % 5000 == 0:
                                print ("Vong lap thu : {0}".format(so_vong_lap))

print ("Processing DONE !!!")
