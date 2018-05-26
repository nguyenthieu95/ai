import sys, os, time
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
from model import script3
from model.utils import IOHelper
from pandas import read_csv

data = [5]
list_number_data = [(6640, 8300, 0)]
number_run_test = 15


finalRMSE = []
finalMAE = []

for loop in range(0, number_run_test):

    arrayRMSE = []
    arrayMAE = []

    for i in range(0, len(data)):
        pathsave = "/home/hunter/nguyenthieu95/ai/do_an/my_neural/script3/test1/result/" + str(data[i]) + "m/ram/"
        fullpath = "/home/hunter/nguyenthieu95/ai/data/GoogleTrace/"
        filename = "data_resource_usage_" + str(data[i]) + "Minutes_6176858948.csv"
        filesave_model = "/home/hunter/nguyenthieu95/ai/do_an/my_neural/script3/test1/result/" + str(data[i]) + "m/ram.txt"
        df = read_csv(fullpath + filename, header=None, index_col=False, usecols=[4], engine='python')
        dataset_original = df.values
        list_num = list_number_data[i]

        output_index = 0                # 0: cpu, 1: ram
        method_statistic = 0
        max_cluster=50
        mutation_id=1
        couple_activation = (2, 0)        # 0: elu, 1:relu, 2:tanh, 3:sigmoid

        sliding_windows = [3]
        positive_numbers = [0.15]
        stimulation_levels = [0.25]
        distance_levels = [0.50]

        list_max_gens = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
        list_num_bees = [80]
        list_couple_bees = [(10, 3)]
        num_sites = 3
        elite_sites = 1
        patch_size = 5.0
        patch_factor = 0.97
        low_up_w = [-1, 1]
        low_up_b = [-1, 1]

        fig_id = 1
        so_vong_lap = 0
        for sliding in sliding_windows:
            for pos_number in positive_numbers:
                for sti_level in stimulation_levels:
                    for dist_level in distance_levels:

                        for max_gens in list_max_gens:
                            for num_bees in list_num_bees:
                                for couple_bees in list_couple_bees:

                                    start_time = time.time()
                                    model_name = "Loop=" + str(loop) + "_sliding=" + str(sliding) + "_posNumber=" + str(pos_number) + \
                                                 "_stiLevel=" + str(sti_level) + "_disLevel=" + str(dist_level) + \
                                                 "_maxGens=" + str(max_gens) + "_numBees=" + str(num_bees) + "_coupleBees=" + str(couple_bees)

                                    para_data = {
                                        "dataset": dataset_original, "list_index": list_num,
                                        "output_index": output_index, "method_statistic": method_statistic,
                                        "sliding": sliding, "loop": loop
                                    }
                                    para_net = {
                                        "max_cluster": max_cluster, "pos_number": pos_number,
                                        "sti_level": sti_level, "dist_level": dist_level,
                                        "mutation_id": mutation_id, "couple_activation": couple_activation,
                                        "path_save": pathsave, "fig_id": fig_id
                                    }
                                    para_bee = {
                                        "max_gens": max_gens, "num_bees": num_bees,
                                        "num_sites": num_sites, "elite_sites": elite_sites,
                                        "patch_size": patch_size, "patch_factor": patch_factor,
                                        "couple_bees": couple_bees, "lowup_w": low_up_b, "lowup_b": low_up_b
                                    }
                                    my_model = script3.SOBEE(para_data, para_net, para_bee)
                                    my_model.fit()
                                    time_model = round(time.time() - start_time, 3)

                                    arrayRMSE.append(my_model.RMSE)
                                    arrayMAE.append(my_model.MAE)

                                    temp = [my_model.time_cluster, my_model.time_train, time_model]
                                    IOHelper.save_model(my_model.list_clusters, my_model.weight, my_model.bias, temp,
                                                        my_model.RMSE, my_model.MAE, model_name, filesave_model)

                                    so_vong_lap += 1
                                    fig_id += 2
                                    if so_vong_lap % 100 == 0:
                                        print ("Vong lap thu : {0}".format(so_vong_lap))

    print ("Processing loop {0} DONE!!!".format(loop))

    finalRMSE.append(arrayRMSE)
    finalMAE.append(arrayMAE)

filename_runtest = "/home/hunter/nguyenthieu95/ai/do_an/my_neural/script3/test1/result/"
IOHelper.save_run_test(number_run_test, finalMAE, filename_runtest + "ram_MAE.csv")
IOHelper.save_run_test(number_run_test, finalRMSE, filename_runtest + "ram_RMSE.csv")


