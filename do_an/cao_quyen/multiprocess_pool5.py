from multiprocessing import Pool
from queue import Queue

import pandas as pd
from sklearn.model_selection import ParameterGrid
from model.flnn import Model as FLNN
from utils.Setting import param_grid_test as param_grid


# parameters
data_index = [5]
list_idx = [(6640, 0, 8300)]
queue = Queue()

def train_model(item):
    sliding_window = item["sliding_window"]
    expand_func = item["expand_func"]
    activation = item["activation"]
    epoch = item["epoch"]

    learning_rate = item["learning_rate"]
    batch_size = item["batch_size"]
    beta = item["beta"]

    p = FLNN(dataset_original, idx[0], idx[1], idx[2], sliding=sliding_window, activation=activation,
             expand_func=expand_func, epoch=epoch, learning_rate=learning_rate,
             batch_size=batch_size, beta=beta, test_name=test_name,
             path_save_result=path_save_result, method_statistic=method_statistic,
             output_index=output_index, output_multi=output_multi)
    p.train()


#Producer
for index, dataindex in enumerate(data_index):

    # Combination of: usecols, output_index
    # ( [number], None ), ([number], number)   ==> single input, single output
    # ( [number, number, ...], None )   ==> multiple input, multiple output                 ==> output_multi = True
    # ( [number, number, ...], number ) ==> multiple input, single output

    df = pd.read_csv("data/" + 'data_resource_usage_' + str(dataindex) + 'Minutes_6176858948.csv', usecols=[3, 4], header=None, index_col=False)
    df.dropna(inplace=True)

    # parameters
    dataset_original = df.values
    idx = list_idx[index]
    test_name = "tn1"
    path_save_result = "test/" + test_name + "/fl_gann/multi/"
    output_index = None
    output_multi = True
    method_statistic = 0

    # Create combination of params.
    for item in list(ParameterGrid(param_grid)) :
        queue.put_nowait(item)

# Consumer
pool = Pool(8)
pool.map(train_model, list(queue.queue))
pool.close()
pool.join()
pool.terminate()