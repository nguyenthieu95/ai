import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../../")
from model import sonia
from pandas import read_csv

# pathsave = "/home/thieunv/Desktop/Link to LabThayMinh/code/6_google_trace/SVNCKH/testing/3m/sonia/result/cpu/"
# fullpath = "/home/thieunv/university/LabThayMinh/code/data/GoogleTrace/"

pathsave = "/home/hunter/nguyenthieu95/ai/6_google_trace/SVNCKH/testing/10m/sonia/result/ram/"
fullpath = "/home/hunter/nguyenthieu95/ai/data/GoogleTrace/"

filename10 = "data_resource_usage_10Minutes_6176858948.csv"
df = read_csv(fullpath+ filename10, header=None, index_col=False, usecols=[4], engine='python')
dataset_original = df.values

list_num10 = (3280, 4100, 0)
output_index = 0                # 0: cpu, 1: ram
method_statistic = 0
max_cluster=20
mutation_id=1
activation_id= 2            # 0: elu, 1:relu, 2:tanh, 3:sigmoid
activation_id2 = 3

epochs = [480, 1000, 2000]
batch_sizes = [8, 32, 64]
learning_rates = [0.05, 0.15, 0.35]
sliding_windows = [ 2, 3, 5]
positive_numbers = [0.05, 0.15, 0.35]
stimulation_levels = [0.10, 0.25, 0.45]
distance_levels = [0.10, 0.25, 0.45]

fig_id = 1
so_vong_lap = 0
for epoch in epochs:
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for sliding in sliding_windows:
                for positive_number in positive_numbers:
                    for sti_level in stimulation_levels:
                        for dis_level in distance_levels:
                            my_model = sonia.Model(dataset_original, list_num10, output_index, epoch, batch_size, learning_rate, sliding, method_statistic, max_cluster,
                                           sti_level, positive_number, dis_level, mutation_id, activation_id, activation_id2, fig_id, pathsave)
                            my_model.fit()
                            so_vong_lap += 1
                            fig_id += 2
                            if so_vong_lap % 100 == 0:
                                print "Vong lap thu : {0}".format(so_vong_lap)

print "Processing DONE !!!"
