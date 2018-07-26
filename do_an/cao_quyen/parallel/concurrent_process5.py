from concurrent.futures import ProcessPoolExecutor
from queue import Queue

import pandas as pd
from sklearn.model_selection import ParameterGrid
from model.flgann import Model as FLGANN
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
    pop_size = item["pop_size"]
    pc = item["pc"]
    pm = item["pm"]

    p = FLGANN(dataset_original, idx[0], idx[1], idx[2], sliding=sliding_window, activation=activation,
               expand_func=expand_func, epoch=epoch, pop_size=pop_size, pc=pc, pm=pm,
               lowup_w=lowup_w, lowup_b=lowup_b, test_name=test_name,
               path_save_result=path_save_result, method_statistic=method_statistic,
               output_index=output_index, output_multi=output_multi)
    p.run()


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
    lowup_w = [-1, 1]
    lowup_b = [-1, 1]

    # Create combination of params.
    for item in list(ParameterGrid(param_grid)) :
        queue.put_nowait(item)

# Consumer
with ProcessPoolExecutor(max_workers=4) as executor:
    executor.map(train_model, list(queue.queue))

