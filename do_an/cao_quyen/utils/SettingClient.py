### Run on sh file
#pd_readfilepath = "data/"
#path_save_result = "code_run/tn1/fl_gann/"

### Run on each python file
#pd_readfilepath = "../../../data/"
#path_save_result = ""


pd_readfilepath = "data/"
path_save_result = "test/tn1/fl_gann/"


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

param_grid_ga_real_client = {
    "sliding_window": [2, 3, 5],
    "expand_func": [0, 2, 4],         # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": [1, 3],          # 1:elu, 3:tanh,

    "epoch": [600, 700, 800],
    "pop_size": [150, 200, 250],               # 100 -> 900
    "pc": [0.85],               # 0.85 -> 0.97
    "pm": [0.05, 0.15]                 # 0.02 -> 0.1
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






###### TN 2
requirement_variables_tn2_ram = [
    "../../../data/",    # pd.readfilepath
    [4],        # usecols trong pd
    test_name,      # test_name
    "results/" + "ram/",    # path_save_result
    None,       # output_index
    False,      # output_multi
]

requirement_variables_tn2_cpu = [
    "../../../data/",    # pd.readfilepath
    [3],        # usecols trong pd
    test_name,      # test_name
    "results/" + "cpu/",    # path_save_result
    None,       # output_index
    False,      # output_multi
]

param_grid_ga_real_client_tn2_pop_size = {
    "sliding_window": [3],
    "expand_func": [0],         # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": [1],          # 1:elu, 3:tanh,

    "epoch": [600],
    "pop_size": [100, 150, 200, 250, 300, 350, 400, 450, 500],               # 100 -> 900
    "pc": [0.85],               # 0.85 -> 0.97
    "pm": [0.15]                 # 0.02 -> 0.1
}

param_grid_ga_real_client_tn2_pc = {
    "sliding_window": [3],
    "expand_func": [0],         # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": [1],          # 1:elu, 3:tanh,

    "epoch": [600],
    "pop_size": [250],               # 100 -> 900
    "pc": [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98],               # 0.85 -> 0.97
    "pm": [0.15]                 # 0.02 -> 0.1
}

param_grid_ga_real_client_tn2_pm = {
    "sliding_window": [3],
    "expand_func": [0],         # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": [1],          # 1:elu, 3:tanh,

    "epoch": [600],
    "pop_size": [250],               # 100 -> 900
    "pc": [0.85],               # 0.85 -> 0.97
    "pm": [0.02, 0.05, 0.075, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30]                 # 0.02 -> 0.1
}




###### TN 3: Changing functional link

param_grid_ga_real_client_tn3 = {
    "sliding_window": [2, 3, 4, 5],
    "expand_func": [0, 1, 2, 3],         # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": [1],          # 1:elu, 3:tanh,

    "epoch": [600],
    "pop_size": [200],               # 100 -> 900
    "pc": [0.85],               # 0.85 -> 0.97
    "pm": [0.15]                 # 0.02 -> 0.1
}


tn3_readfilepath = "data/"
tn3_savepath = "test/tn3/"
test_name_3 = "tn3"

requirement_variables_tn3_cpu = [
    tn3_readfilepath,    # pd.readfilepath
    [3],        # usecols trong pd
    test_name_3,      # test_name
    tn3_savepath + "cpu/",    # path_save_result
    None,       # output_index
    False,      # output_multi
]

requirement_variables_tn3_multi = [
    tn3_readfilepath,    # pd.readfilepath
    [3, 4],        # usecols trong pd
    test_name_3,      # test_name
    tn3_savepath + "multi/",    # path_save_result
    None,       # output_index
    True,      # output_multi
]

requirement_variables_tn3_multi_cpu = [
    tn3_readfilepath,    # pd.readfilepath
    [3, 4],        # usecols trong pd
    test_name_3,      # test_name
    tn3_savepath + "multi_cpu/",    # path_save_result
    0,       # output_index
    False,      # output_multi
]

requirement_variables_tn3_multi_ram = [
    tn3_readfilepath,    # pd.readfilepath
    [3, 4],        # usecols trong pd
    test_name_3,      # test_name
    tn3_savepath + "multi_ram/",    # path_save_result
    1,       # output_index
    False,      # output_multi
]

requirement_variables_tn3_ram = [
    tn3_readfilepath,    # pd.readfilepath
    [4],        # usecols trong pd
    test_name_3,      # test_name
    tn3_savepath + "ram/",    # path_save_result
    None,       # output_index
    False,      # output_multi
]
