from model.fl_gann import Model as FLGANN
import pandas as pd

# parameters
data_index = [5]
list_idx = [(6640, 0, 8300)]           # [(4980, 1, 8300)]
methods = ['FLGANN']

for index, dataindex in enumerate(data_index):
    for method in methods:

        df = pd.read_csv("data/" + 'data_resource_usage_' + str(dataindex) + 'Minutes_6176858948.csv', usecols=[3], header=None, index_col=False)
        df.dropna(inplace=True)

        # parameters
        dataset_original = df.values
        idx = list_idx[index]
        test_name = "tn1"
        path_save_result = "test/" + test_name + "/flnn/multi/"
        output_index = None
        output_multi = False
        method_statistic = 0
        lowup_w = [-1, 1]
        lowup_b = [-1, 1]

        sliding_windows = [3]
        expand_funcs = [0]  # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries,
        activations = [0, 1, 2, 3, 4]  # 0: self, 1:elu, 2:relu, 3:tanh, 4:sigmoid

        # FL-GANN
        epochs = [800]
        pop_sizes = [200]        # 100 -> 900
        pcs = [0.90]            # 0.85 -> 0.97
        pms = [0.02]            # 0.02 -> 0.1


        if method == 'FLGANN':

            for sw in sliding_windows:
                for expand_func in expand_funcs:
                    for activation in activations:
                        for epoch in epochs:

                            for pop_size in pop_sizes:
                                for pc in pcs:
                                    for pm in pms:

                                        p = FLGANN(dataset_original, idx[0], idx[1], idx[2], sw, activation=activation,
                                                 expand_func=expand_func, epoch=epoch, pop_size=pop_size, pc=pc, pm=pm,
                                                    lowup_w = lowup_w, lowup_b = lowup_b, test_name=test_name,
                                                 path_save_result=path_save_result, method_statistic=method_statistic,
                                                 output_index=output_index, output_multi=output_multi)
                                        p.run()


