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

tn1_requirement_variables_multi = [
    pd_readfilepath,    # pd.readfilepath
    [3, 4],        # usecols trong pd
    test_name,      # test_name
    path_save_result + "multi/",    # path_save_result
    None,       # output_index
    True,      # output_multi
]

tn1_requirement_variables_multi_ram = [
    pd_readfilepath,    # pd.readfilepath
    [3, 4],        # usecols trong pd
    test_name,      # test_name
    path_save_result + "multi_ram/",    # path_save_result
    1,       # output_index
    False,      # output_multi
]

requirement_variables_multi = [
    pd_readfilepath,    # pd.readfilepath
    [3, 4],        # usecols trong pd
    test_name,      # test_name
    "test/tn1/flnn/" + "multi/",    # path_save_result
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
    "test/tn1/flnn/" + "multi_ram/",    # path_save_result
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


flnn_path_save_result = "test/tn1/flnn/"


flnn_requirement_variables_cpu = [
    pd_readfilepath,    # pd.readfilepath
    [3],        # usecols trong pd
    test_name,      # test_name
    flnn_path_save_result + "cpu/",    # path_save_result
    None,       # output_index
    False,      # output_multi
]

flnn_requirement_variables_multi_cpu = [
    pd_readfilepath,    # pd.readfilepath
    [3, 4],        # usecols trong pd
    test_name,      # test_name
    flnn_path_save_result + "multi_cpu/",    # path_save_result
    0,       # output_index
    False,      # output_multi
]

flnn_requirement_variables_ram = [
    pd_readfilepath,    # pd.readfilepath
    [4],        # usecols trong pd
    test_name,      # test_name
    flnn_path_save_result + "ram/",    # path_save_result
    None,       # output_index
    False,      # output_multi
]



##### Settings parameters
client_tn1_param_grid_flnn = {
    "sliding_window": [2],
    "expand_func": [0],         # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": [1],          # 0: self, 1:elu, 2:relu, 3:tanh, 4:sigmoid

    "epoch": [500],
    "learning_rate": [0.05],
    "batch_size": [64],
    "beta": [0.85]
}

server_tn1_param_grid_flnn = {
    "sliding_window": [2, 3, 4, 5],
    "expand_func": [0, 1, 2, 3, 4],         # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": [1],          # 0: self, 1:elu, 2:relu, 3:tanh, 4:sigmoid

    "epoch": [1500],
    "learning_rate": [0.05, 0.15],
    "batch_size": [16, 64],
    "beta": [0.85, 0.90]
}


server_tn1_param_grid_ga = {
    "sliding_window": [2, 3, 4, 5],
    "expand_func": [0, 1, 2, 3, 4],         # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": [1],          # 0: self, 1:elu, 2:relu, 3:tanh, 4:sigmoid

    "epoch": [650, 800],
    "pop_size": [250, 300],               # 100 -> 900
    "pc": [0.85, 0.90],               # 0.85 -> 0.97
    "pm": [0.02, 0.05]                 # 0.02 -> 0.1
}



###### TN 2

test_tn2 = "tn2"
server_tn2_requirement_variables_cpu_ps = [
    "data/",    # pd.readfilepath
    [3],        # usecols trong pd
    test_tn2,      # test_name
    "test/tn2/ps/results/",    # path_save_result
    None,       # output_index
    False,      # output_multi
]

server_tn2_requirement_variables_cpu_pc = [
    "data/",    # pd.readfilepath
    [3],        # usecols trong pd
    test_tn2,      # test_name
    "test/tn2/pc/results/",    # path_save_result
    None,       # output_index
    False,      # output_multi
]

server_tn2_requirement_variables_cpu_pm = [
    "data/",    # pd.readfilepath
    [3],        # usecols trong pd
    test_tn2,      # test_name
    "test/tn2/pm/results/",    # path_save_result
    None,       # output_index
    False,      # output_multi
]




server_tn2_requirement_variables_multi_cpu_ps = [
    "data/",    # pd.readfilepath
    [3, 4],        # usecols trong pd
    test_tn2,      # test_name
    "test/tn2/ps/results/",    # path_save_result
    0,       # output_index
    False,      # output_multi
]

server_tn2_requirement_variables_multi_cpu_pc = [
    "data/",    # pd.readfilepath
    [3, 4],        # usecols trong pd
    test_tn2,      # test_name
    "test/tn2/ps/results/",    # path_save_result
    0,       # output_index
    False,      # output_multi
]

server_tn2_requirement_variables_multi_cpu_pm = [
    "data/",    # pd.readfilepath
    [3, 4],        # usecols trong pd
    test_tn2,      # test_name
    "test/tn2/ps/results/",    # path_save_result
    0,       # output_index
    False,      # output_multi
]

server_tn2_requirement_variables_multi_cpu_function = [
    "data/",    # pd.readfilepath
    [3, 4],        # usecols trong pd
    test_tn2,      # test_name
    "test/tn2/function/results/",    # path_save_result
    0,       # output_index
    False,      # output_multi
]

server_tn2_param_grid_ga_function = {
    "sliding_window": [3],
    "expand_func": [0, 1, 2, 3, 4],         # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": [1],          # 1:elu, 3:tanh,

    "epoch": [650],
    "pop_size": [250],               # 100 -> 900
    "pc": [0.90],               # 0.85 -> 0.97
    "pm": [0.035]                 # 0.02 -> 0.1
}






server_tn2_param_grid_ga_ps = {
    "sliding_window": [3],
    "expand_func": [0],         # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": [1],          # 1:elu, 3:tanh,

    "epoch": [600],
    "pop_size": [100, 150, 200, 250, 300, 350, 400, 450, 500],               # 100 -> 900
    "pc": [0.85],               # 0.85 -> 0.97
    "pm": [0.02]                 # 0.02 -> 0.1
}

server_tn2_param_grid_ga_pc= {
    "sliding_window": [3],
    "expand_func": [0],         # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": [1],          # 1:elu, 3:tanh,

    "epoch": [600],
    "pop_size": [250],               # 100 -> 900
    "pc": [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98],               # 0.85 -> 0.97
    "pm": [0.02]                 # 0.02 -> 0.1
}

server_tn2_param_grid_ga_pm = {
    "sliding_window": [3],
    "expand_func": [0],         # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": [1],          # 1:elu, 3:tanh,

    "epoch": [600],
    "pop_size": [250],               # 100 -> 900
    "pc": [0.85],               # 0.85 -> 0.97
    "pm": [0.035, 0.065, 0.08, 0.09]                # 0.02 -> 0.1

}

#"pm": [0.005, 0.01, 0.02, 0.05, 0.10, 0.125, 0.15, 0.175, 0.20]



###### TN 3: Changing functional link

server_tn3_param_grid_ga = {
    "sliding_window": [2, 3, 4, 5],
    "expand_func": [0, 1, 2, 3, 4],         # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries, 4:trigonometric
    "activation": [1],          # 1:elu, 3:tanh,

    "epoch": [600],
    "pop_size": [200],               # 100 -> 900
    "pc": [0.85],               # 0.85 -> 0.97
    "pm": [0.02]                 # 0.02 -> 0.1
}


tn3_readfilepath = "data/"
tn3_savepath = "test/tn3/"
test_name_3 = "tn3"

server_tn3_requirement_variables_ga_cpu = [
    tn3_readfilepath,    # pd.readfilepath
    [3],        # usecols trong pd
    test_name_3,      # test_name
    tn3_savepath + "cpu/",    # path_save_result
    None,       # output_index
    False,      # output_multi
]

server_tn3_requirement_variables_ga_ram = [
    tn3_readfilepath,    # pd.readfilepath
    [4],        # usecols trong pd
    test_name_3,      # test_name
    tn3_savepath + "ram/",    # path_save_result
    None,       # output_index
    False,      # output_multi
]

server_tn3_requirement_variables_ga_multi_cpu = [
    tn3_readfilepath,    # pd.readfilepath
    [3, 4],        # usecols trong pd
    test_name_3,      # test_name
    tn3_savepath + "multi_cpu/",    # path_save_result
    0,       # output_index
    False,      # output_multi
]

server_tn3_requirement_variables_ga_multi_ram = [
    tn3_readfilepath,    # pd.readfilepath
    [3, 4],        # usecols trong pd
    test_name_3,      # test_name
    tn3_savepath + "multi_ram/",    # path_save_result
    1,       # output_index
    False,      # output_multi
]

server_tn3_requirement_variables_ga_multi = [
    tn3_readfilepath,    # pd.readfilepath
    [3, 4],        # usecols trong pd
    test_name_3,      # test_name
    tn3_savepath + "multi/",    # path_save_result
    None,       # output_index
    True,      # output_multi
]



