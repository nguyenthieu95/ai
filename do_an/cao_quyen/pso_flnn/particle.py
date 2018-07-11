from sklearn.metrics import mean_absolute_error
from utils.MathUtil import *
class Particle:
    def __init__(self, n, activation = 0):
        self.n = n
        self.x = np.random.uniform(low = -1, high = 1, size=(n + 1, 1))
        self.v = np.zeros((n + 1, 1))
        self.pbest = np.copy(self.x)
        self.best_fitness = -1
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
        x, b = self.x[:-1, :], self.x[[-1], :]
        z = np.dot(X, x) + b

        a = self.activation_output(z)

        mae = mean_absolute_error(y, a)
        self.fitness = 1.0 / mae
        return self.fitness

    def predict(self, X):
        x, b = self.x[:-1, :], self.x[[-1], :]
        z = np.dot(X, x) + b
        a = self.activation_output(z)
        return a
