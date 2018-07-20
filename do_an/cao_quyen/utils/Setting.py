### Run on sh file
pd_readfilepath = "data/"
path_save_result = "code_run/tn1/fl_gann/"

### Run on each python file
#pd_readfilepath = "../../../data/"
#path_save_result = ""


test_name = "tn1"

requirement_variables_cpu = [
    pd_readfilepath,    # pd.readfilepath
    [3],        # usecols trong pd
    test_name,      # test_name
    path_save_result + "cpu/",    # path_save_result
    None,       # output_index
    False,      # output_multi
]

requirement_variables_multi = [
    pd_readfilepath,    # pd.readfilepath
    [3, 4],        # usecols trong pd
    test_name,      # test_name
    path_save_result + "multi/",    # path_save_result
    None,       # output_index
    True,      # output_multi
]

requirement_variables_multi_cpu = [
    pd_readfilepath,    # pd.readfilepath
    [3, 4],        # usecols trong pd
    test_name,      # test_name
    path_save_result + "multi_cpu/",    # path_save_result
    0,       # output_index
    False,      # output_multi
]

requirement_variables_multi_ram = [
    pd_readfilepath,    # pd.readfilepath
    [3, 4],        # usecols trong pd
    test_name,      # test_name
    path_save_result + "multi_ram/",    # path_save_result
    1,       # output_index
    False,      # output_multi
]

requirement_variables_ram = [
    pd_readfilepath,    # pd.readfilepath
    [4],        # usecols trong pd
    test_name,      # test_name
    path_save_result + "ram/",    # path_save_result
    None,       # output_index
    False,      # output_multi
]


##### Settings parameters
param_grid_ga_real = {
    "sliding_window": [2, 3, 5],
    "expand_func": [0, 1, 2, 3, 4],         # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": [1, 2, 3, 4],          # 0: self, 1:elu, 2:relu, 3:tanh, 4:sigmoid

    "epoch": [500, 600, 700, 800],
    "pop_size": [100, 150, 200, 250],               # 100 -> 900
    "pc": [0.85, 0.90, 0.95],               # 0.85 -> 0.97
    "pm": [0.05, 0.1, 0.15]                 # 0.02 -> 0.1
}



param_grid_test = {
    "sliding_window": [2, 3, 5],
    "expand_func": [0],         # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": [1, 2, 3],          # 0: self, 1:elu, 2:relu, 3:tanh, 4:sigmoid

    "epoch": [500],
    "pop_size": [100],          # 100 -> 900
    "pc": [0.87],               # 0.85 -> 0.97
    "pm": [0.02, 0.05]                 # 0.02 -> 0.1
}


param_grid_flnn_test = {
    "sliding_window": [2, 3, 5],
    "expand_func": [0],         # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": [1, 2, 3],          # 0: self, 1:elu, 2:relu, 3:tanh, 4:sigmoid

    "epoch": [500],
    "learning_rate": [0.15],
    "batch_size": [64],
    "beta": [0.75, 0.80, 0.85]
}


requirement_variables_test = [
    "data/",    # pd.readfilepath
    [3],        # usecols trong pd
    test_name,      # test_name
    "test/tn1/fl_gann/cpu/",    # path_save_result
    None,       # output_index
    False,      # output_multi
]

requirement_variables_test3 = [
    "data/",    # pd.readfilepath
    [4],        # usecols trong pd
    test_name,      # test_name
    "test/tn1/fl_gann/ram/",    # path_save_result
    None,       # output_index
    False,      # output_multi
]