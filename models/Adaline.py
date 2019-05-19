import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as a3d

class Adaline(object):
    realization_errors, data_set, weights = [], [], []

    def __init__(self, learn_rate, max_epochs):
        self.learn_rate = learn_rate
        self.max_epochs = max_epochs

    @staticmethod
    def y(x, a, b):
        return (a*x) + b
    
    @staticmethod
    def z(x, y, a, b, c):
        return a*x + b*y + c
    
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
    
    @staticmethod
    def plot_3d(dataset, w):
        z_surface = []
        for data in dataset:
            z_surface.append(np.dot(np.array([-1.0, data[1], data[2]]), w))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = list(dataset[:, 1])
        y = list(dataset[:, 2])
        z = list(dataset[:, 0])
        ax.plot_trisurf(x, y, z_surface, linewidth=0.2, antialiased=True)
        ax.scatter(x, y, z, c='k', marker='o')
        plt.show()

    def start_artificial_dataset_2d(self, a, b):
        self.init_weights(2)
        X = np.linspace(0, 10, 100)
        Y = [self.y(x, a, b) + np.random.uniform(-1, 1) for x in X]
        data = np.array([[i, j] for i, j in zip(X, Y)])
        self.data_set = self.normalize(data)

    def start_artificial_dataset_3d(self, a, b, c):
        self.init_weights(3)
        X = np.linspace(0, 10, 100)
        Y = np.random.rand(100, 1)
        Z = [self.z(x, y, a, b, c) + np.random.uniform(-1, 1) for x, y in zip(X, Y)]
        data = np.array([[Z[i][0], X[i], Y[i][0]] for i in range(len(Z))])
        self.data_set = self.normalize(data)
        print(self.data_set)
        #self.plot_3d(self.data_set, 0)


    def train(self, dataset):
        self.shuflle(dataset)
        qt_trainning = int(0.8 * len(dataset))
        self.train_data, self.test_data = dataset[:qt_trainning], dataset[qt_trainning:]

        for _ in range(self.max_epochs):
            self.shuflle(self.train_data)
            for point in self.train_data:
                x = point[0]
                y = point[1]
                inputs = np.array([-1.0, x])
                guess = np.dot(inputs, self.weights)
                error = y - guess
                self.weights += self.learn_rate * error * inputs

    def train_3d(self, dataset):
        self.shuflle(dataset)
        qt_trainning = int(0.8 * len(dataset))
        self.train_data, self.test_data = dataset[:qt_trainning], dataset[qt_trainning:]

        for _ in range(self.max_epochs):
            self.shuflle(self.train_data)
            for point in self.train_data:
                z = point[0]
                x = point[1]
                y = point[2]
                inputs = np.array([-1.0, x, y])
                guess = np.dot(inputs, self.weights)
                error = z - guess
                self.weights += self.learn_rate * error * inputs
        print(self.weights)

    def test(self):
        errors = []
        for data in self.test_data:
            y = np.dot(np.array([-1.0, data[0]]), self.weights)
            error = data[1] - y
            errors.append(error * error)
        self.realization_errors.append(np.mean(errors))

    def init_weights(self, size):
        self.weights = np.random.rand(size)
