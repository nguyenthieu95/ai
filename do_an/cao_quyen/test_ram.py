from ga_flnn.model import Model as GAModel
from flnn.model import Model as FLNN
import pandas as pd

# parameters
data_index = [5]
list_idx = [(6640, 1660)]
methods = ['FLNN', 'GA']

for index, dataindex in enumerate(data_index):
    for method in methods:

        df = pd.read_csv("data/" + 'data_resource_usage_' + str(dataindex) + 'Minutes_6176858948.csv', usecols=[4], header=None, index_col=False)
        df.dropna(inplace=True)

        # parameters
        dataset_original = df.values
        idx = list_idx[index]
        test_name = "tn1"
        path_save_result = "test/" + test_name + "/ram/"

        sliding_windows = [2, 3, 4, 5]
        expand_funcs = [0, 1, 2, 3 ]  # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries,
        activations = [0, 1, 2, 3]  # 0: self, 1:elu, 2:relu, 3:tanh, 4:sigmoid

        # FLNN
        epoch_flnns = [800, 1200, 1500, 2000]
        learning_rates = [0.05, 0.10, 0.15, 0.25]
        batch_sizes = [16, 32, 64, 256]
        betas = [0.80, 0.85, 0.90]       # momemtum 0.7 -> 0.9 best

        # GA
        epoch_flganns = [650, 800, 1000, 1200]
        pop_sizes = [100, 130, 170, 200]
        pcs = [0.7, 0.75, 0.8, 0.85]
        pms = [0.005, 0.01, 0.015, 0.02]

        if method == 'FLNN':

            for sw in sliding_windows:
                for expand_func in expand_funcs:
                    for activation in activations:
                        for epoch in epoch_flnns:

                            for learning_rate in learning_rates:
                                for batch_size in batch_sizes:
                                    for beta in betas:

                                        p = FLNN(dataset_original, idx[0], idx[1], sw, activation=activation,
                                                 expand_func=expand_func, epoch=epoch, learning_rate=learning_rate,
                                                 batch_size=batch_size,
                                                 beta=beta, test_name=test_name, path_save_result=path_save_result)
                                        p.train()

        if method == 'GA':

            for sw in sliding_windows:
                for expand_func in expand_funcs:
                    for activation in activations:
                        for epoch in epoch_flganns:

                            for pop_size in pop_sizes:
                                for pc in pcs:
                                    for pm in pms:

                                        p = GAModel(dataset_original, idx[0], idx[1], sw, expand_func=expand_func, epoch=epoch, pop_size=pop_size, pc=pc,
                                                    pm=pm, activation=activation, test_name=test_name, path_save_result=path_save_result)
                                        p.train()






