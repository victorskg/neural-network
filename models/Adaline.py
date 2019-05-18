import numpy as np
import matplotlib.pyplot as plt

class Adaline(object):
    realization_errors, data_set, weights = [], [], []

    def __init__(self, learn_rate, max_epochs, required_precision):
        self.learn_rate = learn_rate
        self.max_epochs = max_epochs
        self.required_precision = required_precision

    @staticmethod
    def y(x, a, b):
        return (a*x) + b
    
    @staticmethod
    def shuflle(values):
        np.random.shuffle(values)

    @staticmethod
    def normalize(dataset):
        for i in range(dataset.shape[1]):
            max_ = max(dataset[:, i])
            min_ = min(dataset[:, i])
            for j in range(dataset.shape[0]):
                dataset[j, i] = (dataset[j, i] - min_) / (max_ - min_)
        return dataset
    
    @staticmethod
    def plot(dataset, w):
        y = []
        for data in dataset:
            y.append(np.dot(np.array([-1.0, data[0]]), w))
        
        plt.scatter(dataset[:, 0], dataset[:, 1], s=3, c='r')
        plt.plot(dataset[:, 0], y, color='b')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('ADALINE - Result after training')
        plt.show()

    def start_artificial_dataset(self, a, b):
        self.init_weights()
        X = np.linspace(0, 10, 100)
        Y = [self.y(x, a, b) + np.random.uniform(-1, 1) for x in X]
        data = [[i, j] for i, j in zip(X, Y)]
        self.data_set = self.normalize(np.array(data))

    def train(self, dataset):
        self.shuflle(dataset)
        qt_trainning = int(0.8 * len(dataset))
        self.train_data, self.test_data = dataset[:qt_trainning], dataset[qt_trainning:]

        for epoch in range(self.max_epochs):
            self.shuflle(self.train_data)
            for point in self.train_data:
                x = point[0]
                y = point[1]
                inputs = np.array([-1.0, x])
                guess = np.dot(inputs, self.weights)
                error = y - guess
                self.weights += self.learn_rate * error * inputs

    def test(self):
        errors = []
        for data in self.test_data:
            y = np.dot(np.array([-1.0, data[0]]), self.weights)
            error = data[1] - y
            errors.append(error * error)
        self.realization_errors.append(np.mean(errors))

    def init_weights(self):
        self.weights = np.random.rand(2)
