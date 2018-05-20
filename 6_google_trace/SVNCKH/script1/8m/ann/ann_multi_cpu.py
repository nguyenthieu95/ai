import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../../")
from model import ann
from pandas import read_csv

# pathsave = "/home/thieunv/Desktop/Link to LabThayMinh/code/6_google_trace/SVNCKH/testing/3m/ann/result/cpu/"
# fullpath = "/home/thieunv/university/LabThayMinh/code/data/GoogleTrace/"

pathsave = "/home/ubuntu/nguyenthieu95/ai/6_google_trace/SVNCKH/testing/8m/ann/result/multi_cpu/"
fullpath = "/home/ubuntu/nguyenthieu95/ai/data/GoogleTrace/"

filename8 = "data_resource_usage_8Minutes_6176858948.csv"
df = read_csv(fullpath+ filename8, header=None, index_col=False, usecols=[3, 4], engine='python')
dataset_original = df.values

list_num8 = (4160, 5200, 0)
output_index = 0
method_statistic = 0

couple_acts = [(0, 0), (0, 3), (1, 0), (1, 3), (2, 2), (3, 3), (3, 0)]      # 0: elu, 1:relu, 2:tanh, 3:sigmoid
optimizers =  [0, 1, 2, 3]
learning_rates =  [0.05, 0.15, 0.35]
num_hiddens = [8, 12, 15]
sliding_windows = [ 2, 3, 5]
epochs =  [800, 1300, 2200]
batch_sizes = [8, 32, 128]


fig_id = 1
so_vong_lap = 0
for epoch in epochs:
    for batch_size in batch_sizes:
        for sliding in sliding_windows:
            for num_hidden in num_hiddens:
                for couple_act in couple_acts:
                    for optimizer in optimizers:
                        for learning_rate in learning_rates:

                            my_model = ann.Model(dataset_original, list_num8, epoch, batch_size, sliding, method_statistic, output_index, num_hidden,
                                             learning_rate, couple_act, optimizer, pathsave, fig_id)
                            my_model.fit()
                            so_vong_lap += 1
                            fig_id += 2
                            if so_vong_lap % 100 == 0:
                                print "Vong lap thu : {0}".format(so_vong_lap)
print "Processing DONE !!!"
