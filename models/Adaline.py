import numpy as np
import matplotlib.pyplot as plt

class Adaline(object):
    data_set, weights = [], []

    def __init__(self, learn_rate, max_epochs, required_precision, a=2, b=3):
        self.learn_rate = learn_rate
        self.max_epochs = max_epochs
        self.required_precision = required_precision
        self.start_artificial_dataset(a, b)
        self.init_weights()
        print(self.weights)
        plt.scatter(self.data_set[0], self.data_set[1], c='g')
        plt.show()

    @staticmethod
    def y(x, a, b):
        return (a*x) + b

    def start_artificial_dataset(self, a, b):
        X = np.linspace(0, 10, 100)
        Y = [self.y(x, a, b) + np.random.rand() * 2 for x in X]
        self.data_set.append(X)
        self.data_set.append(Y)

    def init_weights(self):
        self.weights = np.random.rand(2)