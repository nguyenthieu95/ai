from multiprocessing import JoinableQueue,Queue, Process,cpu_count
from queue import Empty
import time
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid

from ga_flnn.model import Model as GAModel
from flnn.flnn import FLNN


# parameters
data_index = [5]
list_idx = [(6640, 1660)]
methods = ['FLNN', 'GA']
queue = Queue()
# Consumer
WORKER_POOLS = cpu_count()
WORKERS = []
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def train_model(queue,name):
    LOGGER.info("Start worker %s",name)
    while True:
        try:
            item = queue.get(True,timeout=1)
            pop_size = item["pop_size"]
            pc = item["pc"]
            pm = item["pm"]
            method = item["method"]

            if method == 'GA':
                p = GAModel(dataset_original, idx[0], idx[1], sw, expand_func=expand_func, pop_size=pop_size, pc=pc,
                        pm=pm, activation=activation, test_name = test_name, path_save_result = path_save_result)
                p.train(epochs=epoch)
            elif method == 'FLNN':
                p = FLNN(dataset_original, idx[0], idx[0] + idx[1], sw, activation=activation, e_func=expand_func,
                        learning_rate=learning_rate, batch_size=batch_size, beta=beta, test_name = test_name, path_save_result = path_save_result)
                p.train(epochs=epoch)
        except Empty as e:
            break
#Producer

for index, dataindex in enumerate(data_index):

    df = pd.read_csv("data/" + 'data_resource_usage_' + str(dataindex) + 'Minutes_6176858948.csv', usecols=[3], header=None, index_col=False)
    df.dropna(inplace=True)

    # parameters
    dataset_original = df.values
    idx = list_idx[index]

    sliding_windows = [2]
    expand_func = 0  # 0:chebyshev, 1:powerseries, 2:laguerre, 3:legendre
    activation = 3  # 0: self, 1:tanh, 2:relu, 3:elu
    test_name = "tn1"
    path_save_result = "test/" + test_name + "/"
    epoch = 400

    # FLNN
    learning_rate = 0.15
    batch_size = 64
    beta = 0.05

    # GA
    pop_size = np.arange(100,1000,step=10)
    pc = np.arange(0.1,0.5,step=0.2)
    pm = np.arange(0,1,1.0,step=0.1)

    param_grid = {
        "pop_size": pop_size,
        "pc": pc,
        "pm": pm,
        "method": methods
    }
    # Create combination of params.
    for item in list(ParameterGrid(param_grid)) :
        queue.put_nowait(item)


for i in range(WORKER_POOLS):
    WORKERS.append(Process(target=train_model,args=(queue,"worker_%s"%i)))

for i in range(WORKER_POOLS):    
    WORKERS[i].start()
time.sleep(5)
for i in range(WORKER_POOLS):
    WORKERS[i].join()
    LOGGER.info("Terminate workers")
    WORKERS[i].terminate()
