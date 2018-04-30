import numpy as np

def load_data(data_file):
    fp = open(data_file)
    line = fp.readline()
    line = fp.readline().strip()
    input = list()
    output = list()
    while line:
        arr = line.split(',')
        if arr[1] == '':
            arr[1] = 0
        output.append(float(arr[1]))
        input_parse = arr[0].split()
        day = input_parse[0].split('-')
        hour = input_parse[1].split(':')
        input.append([day[1], day[2], hour[0], hour[1]])
        line = fp.readline().strip()
    fp.close()
    train_data = (np.array(input[:72151]), np.array(output[:72151]))
    test_data = (np.array(input[72151:]), np.array(output[72151:]))
    return train_data, test_data

train_data, test_data = load_data('wc98_workload_min.csv')

print train_data


