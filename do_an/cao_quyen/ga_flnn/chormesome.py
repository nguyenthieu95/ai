from sklearn.metrics import mean_absolute_error
from utils.MathUtil import *

class Chormesome:
    def __init__(self, n, w = None, activation = 0):
        self.w = np.random.uniform(low = -1, high = 1, size=(n + 1, 1))

        if w is not None:
            self.w = w
        self.activation = activation

    def activation_output(self, z):
        if self.activation == 0:
            a = z
        elif self.activation == 1:
            a = elu(z)
        elif self.activation == 2:
            a = relu(z)
        elif self.activation == 3:    
            a = tanh(z)
        elif self.activation == 4:
            a = sigmoid(z)
        return a

    def compute_fitness(self, X, y):
        w, b = self.w[:-1, :], self.w[[-1], :]
        z = np.dot(X, w) + b
        a = self.activation_output(z)
        mae = mean_absolute_error(y, a)

        self.fitness = 1 / mae
        return self.fitness

    def predict(self, X):
        w, b = self.w[:-1, :], self.w[[-1], :]
        z = np.dot(X, w) + b
        a = self.activation_output(z)
        return a
