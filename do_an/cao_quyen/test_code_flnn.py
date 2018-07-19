from model.flnn import Model as FLNN
import pandas as pd

# parameters
data_index = [5]
list_idx = [(6640, 0, 8300)]
methods = ['FLNN']

for index, dataindex in enumerate(data_index):
    for method in methods:

        df = pd.read_csv("data/" + 'data_resource_usage_' + str(dataindex) + 'Minutes_6176858948.csv', usecols=[3, 4], header=None, index_col=False)
        df.dropna(inplace=True)

        # parameters
        dataset_original = df.values
        idx = list_idx[index]
        test_name = "tn1"
        path_save_result = "test/" + test_name + "/flnn/cpu/"
        output_index = None
        output_multi = True
        method_statistic = 0

        sliding_windows = [3]
        expand_funcs = [0 ]  # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries,
        activations = [1]  # 0: self, 1:elu, 2:relu, 3:tanh, 4:sigmoid

        # FLNN
        epoch_flnns = [100]
        learning_rates = [0.05]
        batch_sizes = [16]
        betas = [0.80]       # momemtum 0.7 -> 0.9 best


        if method == 'FLNN':

            for sw in sliding_windows:
                for expand_func in expand_funcs:
                    for activation in activations:
                        for epoch in epoch_flnns:

                            for learning_rate in learning_rates:
                                for batch_size in batch_sizes:
                                    for beta in betas:

                                        p = FLNN(dataset_original, idx[0], idx[1], idx[2], sw, activation=activation,
                                                 expand_func=expand_func, epoch=epoch, learning_rate=learning_rate,
                                                 batch_size=batch_size, beta=beta, test_name=test_name,
                                                 path_save_result=path_save_result, method_statistic=method_statistic,
                                                 output_index=output_index, output_multi=output_multi)
                                        p.train()



