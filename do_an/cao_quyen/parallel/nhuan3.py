from multiprocessing import Queue, Process,cpu_count
from queue import Empty
import time
import logging
import pandas as pd
from sklearn.model_selection import ParameterGrid
from model.fl_gann import Model as FLGANN
from utils.Setting import requirement_variables_test3 as requirement_variables
from utils.Setting import param_grid_test as param_grid

# parameters
data_index = [5]
list_idx = [(6640, 0, 8300)]
queue = Queue()

# Consumer
WORKER_POOLS = 16
WORKERS = []
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def train_model(queue,name):
    LOGGER.info("Start worker %s",name)
    while True:
        try:
            item = queue.get(True,timeout=1)

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

        except Empty as e:
            break



#Producer
for index, dataindex in enumerate(data_index):

    # Combination of: usecols, output_index
    # ( [number], None ), ([number], number)   ==> single input, single output
    # ( [number, number, ...], None )   ==> multiple input, multiple output                 ==> output_multi = True
    # ( [number, number, ...], number ) ==> multiple input, single output

    df = pd.read_csv(requirement_variables[0] + 'data_resource_usage_' + str(dataindex) + 'Minutes_6176858948.csv',
                     usecols=requirement_variables[1], header=None, index_col=False)
    df.dropna(inplace=True)

    # parameters
    dataset_original = df.values
    idx = list_idx[index]
    test_name = requirement_variables[2]
    path_save_result = requirement_variables[3]
    output_index = requirement_variables[4]
    output_multi = requirement_variables[5]
    method_statistic = 0            # 0: sliding window, 1: mean, 2: min-mean-max, 3: min-median-max
    lowup_w = [-1, 1]
    lowup_b = [-1, 1]


    # Create combination of params.
    for item in list(ParameterGrid(param_grid)):
        queue.put_nowait(item)

# Consumer
for i in range(WORKER_POOLS):
    WORKERS.append(Process(target=train_model,args=(queue,"worker_%s"%i)))

for i in range(WORKER_POOLS):
    WORKERS[i].start()
time.sleep(5)
for i in range(WORKER_POOLS):
    WORKERS[i].join()
    LOGGER.info("Terminate workers")
    WORKERS[i].terminate()


