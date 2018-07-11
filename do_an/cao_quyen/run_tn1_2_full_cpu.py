from ga_flnn.model import Model as GAModel
from flnn.flnn import FLNN
import pandas as pd
from multiprocessing import JoinableQueue, Process
import time
# parameters
data_index = [5]
list_idx = [(6640, 1660)]
methods = ['FLNN', 'GA']
queue = JoinableQueue()
# Consumer
WORKER_POOLS = 6
WORKERS = []
is_finished = False

def train_model(queue,is_finished):
    while not queue.empty():
        items = queue.get()
        sw = items[0]
        method = items[1]
        if method == 'GA':
            p = GAModel(dataset_original, idx[0], idx[1], sw, expand_func=expand_func, pop_size=pop_size, pc=pc,
                    pm=pm, activation=activation, test_name = test_name, path_save_result = path_save_result)
            p.train(epochs=epoch)
        elif method == 'FLNN':
            p = FLNN(dataset_original, idx[0], idx[0] + idx[1], sw, activation=activation, e_func=expand_func,
                    learning_rate=learning_rate, batch_size=batch_size, beta=beta, test_name = test_name, path_save_result = path_save_result)
            p.train(epochs=epoch)
    is_finished = True
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
    pop_size = 100
    pc = 0.8
    pm = 0.02
    hyper_params = []

    for method in methods:
        for sw in sliding_windows:
            hyper_params.append([method,sw])
            queue.put_nowait([method,sw])


for i in range(WORKER_POOLS):
    WORKERS[i] = Process(target=train_model,args=(queue,is_finished))
    WORKERS[i].start()

while not is_finished:
    time.sleep(1)
for i in range(WORKER_POOLS):
    WORKERS[i].terminate()
