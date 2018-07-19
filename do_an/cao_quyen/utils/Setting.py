pd_readfilepath = "../../../data/"
path_save_result = ""
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
param_grid = {
    "sliding_window": [2],
    "expand_func": [0],         # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": [1],          # 0: self, 1:elu, 2:relu, 3:tanh, 4:sigmoid

    "epoch": [500],
    "pop_size": [100],          # 100 -> 900
    "pc": [0.87],               # 0.85 -> 0.97
    "pm": [0.1]                 # 0.02 -> 0.1
}