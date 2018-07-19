from model.fl_psonn import Model as FLPSONN
import pandas as pd

# parameters
data_index = [5]
list_idx = [(4980, 1, 8300)]          # [(6640, 0, 8300)]

for index, dataindex in enumerate(data_index):

    df = pd.read_csv("data/" + 'data_resource_usage_' + str(dataindex) + 'Minutes_6176858948.csv', usecols=[3], header=None, index_col=False)
    df.dropna(inplace=True)

    # parameters
    dataset_original = df.values
    idx = list_idx[index]
    test_name = "tn1"
    path_save_result = "test/" + test_name + "/fl_psonn/cpu/"
    output_index = None
    output_multi = False
    method_statistic = 0
    lowup_w = [-1, 1]
    lowup_b = [-1, 1]

    sliding_windows = [3]   # 2, 3, 4, 5
    expand_funcs = [0]  # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries,
    activations = [1]  # 0: self, 1:elu, 2:relu, 3:tanh, 4:sigmoid

    # FL-GANN
    epochs = [800]
    pop_sizes = [200]        # 100 -> 900
    train_valid_rates = [(0.4, 0.6)]            # calculate based on Train and Valid dataset with rate

    w_minmaxs = [(0.4, 0.9)]  # [0-1] -> [0.4-0.9]      Trong luong cua con chim
    c_couples = [(0.8, 1.6)]  # [(1.2, 1.2), (0.8, 2.0), (1.6, 0.6)]         #c1, c2 = 2, 2       # [0-2]   Muc do anh huong cua local va global
    # r1, r2 : random theo tung vong lap
    # delta(t) = 1 (do do: x(sau) = x(truoc) + van_toc

    for sw in sliding_windows:
        for expand_func in expand_funcs:
            for activation in activations:
                for epoch in epochs:

                    for pop_size in pop_sizes:
                        for w_minmax in w_minmaxs:
                            for c_couple in c_couples:
                                for train_valid_rate in train_valid_rates:

                                    p = FLPSONN(dataset_original, idx[0], idx[1], idx[2], train_valid_rate=train_valid_rate,
                                                sliding=sw, activation=activation, expand_func=expand_func, epoch=epoch,
                                                pop_size=pop_size, c_couple=c_couple, w_minmax=w_minmax, lowup_w = lowup_w,
                                                lowup_b = lowup_b, test_name=test_name, path_save_result=path_save_result,
                                                method_statistic=method_statistic, output_index=output_index, output_multi=output_multi)
                                    p.run()

