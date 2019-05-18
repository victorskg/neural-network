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
        #plt.scatter(self.data_set[0], self.data_set[1], c='g')
        #plt.show()

    @staticmethod
    def y(x, a, b):
        return (a*x) + b
    
    @staticmethod
    def shuflle(values):
        np.random.shuffle(values)

    def start_artificial_dataset(self, a, b):
        X = np.linspace(0, 10, 100)
        Y = [self.y(x, a, b) + np.random.rand() * 2 for x in X]
        self.data_set.append(X)
        self.data_set.append(Y)

    def train(self, dataset):
        index_iteraction = [i for i in range(100)]
        qt_trainning = int(0.8 * len(index_iteraction))
        self.shuflle(index_iteraction)
        self.train_data, self.test_data = index_iteraction[:qt_trainning], index_iteraction[qt_trainning:]

        for epoch in range(self.max_epochs):
            self.shuflle(self.train_data)
            for index in self.train_data:
                x = self.data_set[0][index]
                y = self.data_set[1][index]
                inputs = np.array([x, -1.0])
                guess = np.dot(inputs, self.weights)
                error = y - guess
                print(self.weights, error, inputs)
                self.weights += self.learn_rate * error * inputs
        print(self.weights)


    def init_weights(self):
        self.weights = np.random.rand(2)