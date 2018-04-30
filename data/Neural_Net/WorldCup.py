import numpy as np

from keras.layers import Dense, Input
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam

def load_data(data_file):
    fp = open(data_file)
    line = fp.readline()
    line = fp.readline().strip()
    input = list()
    output = list()
    while line:
        arr = line.split(',')       # "1998-06-30 19:53:00,137451.0"
        if arr[1] == '':
            arr[1] = 0
        output.append(float(arr[1]))        # "137451.0"
        input_parse = arr[0].split()        # [1998-06-30, 19:53:00]
        day = input_parse[0].split('-')     # [1998, 06, 30]
        hour = input_parse[1].split(':')    # [19, 53, 00]
        input.append([day[1], day[2], hour[0], hour[1]])    # [06, 30, 19, 53]
        line = fp.readline().strip()
    fp.close()
    train_data = (np.array(input[:72151]), np.array(output[:72151]))
    test_data = (np.array(input[72151:]), np.array(output[72151:]))
    return train_data, test_data

def get_model(input_dim, layers, reg_layers):
    input = Input(shape=(input_dim,), dtype='float32', name='input')
    num_layers = len(layers)
    vector = input
    for i in range(num_layers-1):
        layer = Dense(layers[i], activation='relu', W_regularizer=l2(reg_layers[i]), name='layer%d' %i)
        vector = layer(vector)
    # Output layer
    output = Dense(layers[num_layers-1], activation='relu', init='lecun_uniform', name='output')(vector)
    model = Model(input=input, output=output)
    return model

if __name__ == '__main__':
    input_dim = 4
    layers = [32, 32, 32, 32, 32, 1]
    reg_layers = [0.05, 0.05, 0.05, 0.05, 0.05, 1]
    learning_rate = 0.005
    num_epochs = 50

    train_data, test_data = load_data('wc98_workload_min.csv')

    model = get_model(input_dim, layers, reg_layers)
    model.compile(optimizer=Adam(lr=learning_rate), loss='mean_absolute_error', metrics=['accuracy'])
    print model.summary()

    #for epoch in range(num_epochs):
    hist = model.fit(train_data[0], train_data[1], validation_data=test_data,
                     nb_epoch=num_epochs, verbose=2, batch_size=2048, shuffle=True)
    print hist.history['loss']
    print hist.history['val_loss']

    #   1998-06-24 09:16:00,17457.0

    #   1998-06-30 21:44:00,206933.0
    #   1998-06-30 21:56:00,143093.0

    # 1998-06-26 14:30:00,100147.0
    # 1998-06-26 14:15:00,68216.0

    print model.predict(np.array([6, 24, 9, 16]).reshape(1, 4), 2048, 2)

    print model.predict_on_batch(np.array([6, 30, 21, 44]).reshape(1, 4))
    print model.predict_on_batch(np.array([6, 30, 21, 56]).reshape(1, 4))
    print model.predict_on_batch(np.array([6, 26, 14, 30]).reshape(1, 4))
    print model.predict_on_batch(np.array([6, 26, 14, 15]).reshape(1, 4))

