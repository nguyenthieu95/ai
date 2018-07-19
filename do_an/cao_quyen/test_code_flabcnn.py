from model.fl_abcnn import Model as FLABCNN
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
    path_save_result = "test/" + test_name + "/fl_abcnn/cpu/"
    output_index = None
    output_multi = False
    method_statistic = 0

    sliding_windows = [3]   # 2, 3, 4, 5
    expand_funcs = [0]  # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries,
    activations = [1]  # 0: self, 1:elu, 2:relu, 3:tanh, 4:sigmoid

    # FL-ABCNN
    max_gens_list = [800]
    num_bees_list = [200]        # 100 -> 900
    couple_num_bees_list = [(20, 4)]           # e_bees, o_bees
    train_valid_rates = [ (0.4, 0.6) ]       # calculate based on Train and Valid dataset with rate

    patch_variables = (5.0, 0.985)       # patch_size, patch_factor: make patch_size decreased after each epoch
    sites = (3, 1)                      # num_sites : phan vung - thuong la 3 vung, elite_sites : vung tot nhat - thuong la 1
    lowup_w = [-1, 1]
    lowup_b = [-1, 1]

    for sw in sliding_windows:
        for expand_func in expand_funcs:
            for activation in activations:

                for max_gens in max_gens_list:
                    for num_bees in num_bees_list:
                        for couple_num_bees in couple_num_bees_list:

                            for train_valid_rate in train_valid_rates:


                                p = FLABCNN(dataset_original, idx[0], idx[1], idx[2], train_valid_rate, sw, activation=activation,
                                         expand_func=expand_func, max_gens=max_gens, num_bees=num_bees, couple_num_bees=couple_num_bees,
                                        patch_variables=patch_variables, sites=sites, lowup_w = lowup_w, lowup_b = lowup_b,
                                        test_name=test_name, path_save_result=path_save_result, method_statistic=method_statistic,
                                         output_index=output_index, output_multi=output_multi)
                                p.run()
