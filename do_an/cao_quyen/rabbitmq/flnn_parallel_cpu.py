from multiprocessing import Process, cpu_count
from queue import Empty
import time
import logging
import json
import pandas as pd
from sklearn.model_selection import ParameterGrid
from rabbitmq.rabbitmq import RabbitMQClient
from model.flnn import Model as FLNN

# parameters
data_index = [5]
list_idx = [(6640, 0, 8300)]
rabbitmq_client = RabbitMQClient()

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
            item = rabbitmq_client.get_message("test_ml")

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


        except Empty as e:
            break

#Producer
for index, dataindex in enumerate(data_index):

    df = pd.read_csv("data/" + 'data_resource_usage_' + str(dataindex) + 'Minutes_6176858948.csv', usecols=[3], header=None, index_col=False)
    df.dropna(inplace=True)
    # df = None

    # parameters
    dataset_original = df.values
    idx = list_idx[index]
    test_name = "tn1"
    path_save_result = "code_run/" + test_name + "/flnn/cpu/"
    output_index = None
    output_multi = False
    method_statistic = 0

    sliding_window = [2]
    expand_func = [0, 1]  # 0:chebyshev, 1:legendre, 2:laguerre, 3:powerseries,
    activation = [0, 1]  # 0: self, 1:elu, 2:relu, 3:tanh, 4:sigmoid

    # FLNN
    epoch = [100, 200]
    learning_rate = [0.1, 0.2]
    batch_size = [32, 64]
    beta = [0.75]  # momemtum 0.7 -> 0.9 best

    param_grid = {
        "sliding_window": sliding_window,
        "expand_func": expand_func,
        "activation": activation,
        "epoch": epoch,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "beta": beta
    }
    # Create combination of params.
    for cursor, item in enumerate(list(ParameterGrid(param_grid))) :
        print(cursor)
        print(item)
        print(type(item))
        rabbitmq_client.publish_message(exchange_name="test_ml",body=json.dumps(item))


for i in range(WORKER_POOLS):
    WORKERS.append(Process(target=train_model,args=(rabbitmq_client,"worker_%s"%i)))

for i in range(WORKER_POOLS):
    WORKERS[i].start()
time.sleep(5)
for i in range(WORKER_POOLS):
    WORKERS[i].join()
    LOGGER.info("Terminate workers")
    WORKERS[i].terminate()