from sklearn.preprocessing import MinMaxScaler
from utils.MathUtil import *

class ExpandData:
    def __init__(self, data, train_idx, test_idx, sliding, expand_func = None):
        self.data = data
        self.scaler = MinMaxScaler()
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.sliding = sliding
        self.expand_func = expand_func

    def expand_data(self):
        if self.expand_func == 0:
            return chebyshev(self.data)
        elif self.expand_func == 1:
            return legendre(self.data)
        elif self.expand_func == 2:
            return laguerre(self.data)
        elif self.expand_func == 3:
            return powerseries(self.data)

    def scale(self, expanded_data):
        scale_expanded_data = []

        for ed in expanded_data:
            scale_expanded_data.append(self.scaler.fit_transform(ed))

        return scale_expanded_data

    def process_data(self):
        train_idx, test_idx, sliding = self.train_idx, self.test_idx, self.sliding

        expanded_data = self.expand_data()
        scale_data = self.scale(expanded_data)

        data_expanded = np.ones(self.data[:train_idx + test_idx, :].shape)

        for sd in scale_data:
            for i in range(0, sliding):
                data_expanded = np.concatenate((data_expanded, sd[i:i + train_idx + test_idx, :]), axis=1)

        return data_expanded[:, 1:]